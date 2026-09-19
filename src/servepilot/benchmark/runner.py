"""Execute one benchmark (warmup + measured run) against an OpenAI-compatible endpoint."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import replace

from servepilot.benchmark.client import BenchmarkClient
from servepilot.benchmark.gpu_sampler import GPUSampler
from servepilot.benchmark.metrics import aggregate
from servepilot.benchmark.workload import BenchmarkRequest, PromptGenerator, warmup_count
from servepilot.constants import (
    DEFAULT_BENCHMARK_REQUEST_TIMEOUT_SECONDS,
    WARMUP_MAX_REQUESTS,
    WARMUP_MIN_REQUESTS,
)
from servepilot.exceptions import BenchmarkError
from servepilot.hardware.base import HardwareProvider
from servepilot.logging import get_logger
from servepilot.models.tokenizer import TokenCounter
from servepilot.schemas.benchmark import BenchmarkResult, BenchmarkSpec, RequestBenchmarkResult
from servepilot.schemas.workload import WorkloadProfile

log = get_logger(__name__)

ProgressCallback = Callable[[int, int], None]


def make_spec(
    workload: WorkloadProfile,
    *,
    concurrency: int,
    num_requests: int,
    seed: int,
    label: str | None = None,
    request_rate: float | None = None,
    endpoint: str = "chat",
    timeout_seconds: float = DEFAULT_BENCHMARK_REQUEST_TIMEOUT_SECONDS,
) -> BenchmarkSpec:
    return BenchmarkSpec(
        concurrency=concurrency,
        num_requests=num_requests,
        seed=seed,
        streaming=workload.streaming,
        endpoint=endpoint,  # type: ignore[arg-type]
        mode="open" if request_rate else "closed",
        request_rate=request_rate,
        input_tokens_p50=workload.input_tokens_p50,
        input_tokens_p95=workload.input_tokens_p95,
        output_tokens_p50=workload.output_tokens_p50,
        output_tokens_p95=workload.output_tokens_p95,
        warmup_requests=warmup_count(concurrency, WARMUP_MIN_REQUESTS, WARMUP_MAX_REQUESTS),
        request_timeout_seconds=timeout_seconds,
        label=label,
    )


class BenchmarkRunner:
    def __init__(
        self,
        *,
        tokenizer: TokenCounter,
        workload: WorkloadProfile,
        hardware: HardwareProvider | None = None,
        api_key: str | None = None,
        ignore_eos: bool = True,
        trust_server_usage: bool = True,
        requests: Sequence[BenchmarkRequest] | None = None,
        on_observations: Callable[
            [BenchmarkSpec, list[BenchmarkRequest], list[RequestBenchmarkResult]], None
        ]
        | None = None,
    ) -> None:
        self._tok = tokenizer
        self._workload = workload
        self._hardware = hardware
        self._api_key = api_key
        self._ignore_eos = ignore_eos
        self._trust_server_usage = trust_server_usage
        if requests is not None and not requests:
            raise ValueError("replay workload cannot be empty")
        self._requests = list(requests) if requests is not None else None
        self._on_observations = on_observations

    async def _generate(
        self, spec: BenchmarkSpec, count: int, seed_offset: int
    ) -> list[BenchmarkRequest]:
        """Prompt generation tokenizes every request several times; run it off the event loop
        so the router (which shares the loop during tuning) keeps serving."""
        if self._requests is not None:
            source = list(self._requests)
            random.Random(spec.seed + seed_offset).shuffle(source)
            return [replace(source[i % len(source)], index=i) for i in range(count)]
        return await asyncio.to_thread(
            PromptGenerator(self._tok, spec.seed).generate,
            self._workload,
            count,
            seed_offset=seed_offset,
        )

    async def warmup(self, client: BenchmarkClient, spec: BenchmarkSpec) -> None:
        """Send warmup requests and require at least one success before measuring."""
        if spec.warmup_requests <= 0:
            return
        reqs = await self._generate(spec, spec.warmup_requests, 10_000)
        sem = asyncio.Semaphore(max(1, min(spec.concurrency, spec.warmup_requests)))

        async def one(r: BenchmarkRequest) -> RequestBenchmarkResult:
            async with sem:
                return await client.run_request(r, endpoint=spec.endpoint, stream=spec.streaming)

        results = await asyncio.gather(*(one(r) for r in reqs))
        successes = sum(1 for r in results if r.success)
        if successes == 0:
            errors = sorted({r.error or "unknown" for r in results})
            raise BenchmarkError(
                "warmup requests all failed; the endpoint is not generating tokens",
                hints=[f"Backend error: {e}" for e in errors[:3]],
            )
        log.debug("warmup: %d/%d requests succeeded", successes, len(results))

    async def run(
        self,
        base_url: str,
        model: str,
        spec: BenchmarkSpec,
        *,
        candidate_id: str,
        gpu_indices: Sequence[int] = (),
        progress: ProgressCallback | None = None,
        warmup: bool = True,
    ) -> BenchmarkResult:
        requests = await self._generate(spec, spec.num_requests, 0)
        async with BenchmarkClient(
            base_url,
            model=model,
            tokenizer=self._tok,
            timeout_seconds=spec.request_timeout_seconds,
            api_key=self._api_key,
            ignore_eos=self._ignore_eos,
            trust_server_usage=self._trust_server_usage,
        ) as client:
            if warmup:
                await self.warmup(client, spec)
            async with GPUSampler(self._hardware, gpu_indices) as sampler:
                start = time.perf_counter()
                if spec.mode == "open" and spec.request_rate:
                    results = await self._open_loop(client, spec, requests, progress)
                else:
                    results = await self._closed_loop(client, spec, requests, progress)
                duration = time.perf_counter() - start
            if self._on_observations is not None:
                await asyncio.to_thread(self._on_observations, spec, requests, results)
            return aggregate(
                candidate_id,
                spec,
                results,
                gpu_samples=sampler.samples,
                wall_duration_seconds=duration,
            )

    async def _closed_loop(
        self,
        client: BenchmarkClient,
        spec: BenchmarkSpec,
        requests: list[BenchmarkRequest],
        progress: ProgressCallback | None,
    ) -> list[RequestBenchmarkResult]:
        """``concurrency`` workers pull from a shared queue until it is drained."""
        queue: asyncio.Queue[BenchmarkRequest] = asyncio.Queue()
        for r in requests:
            queue.put_nowait(r)
        results: list[RequestBenchmarkResult] = []
        done = 0

        async def worker() -> None:
            nonlocal done
            while True:
                try:
                    r = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                res = await client.run_request(r, endpoint=spec.endpoint, stream=spec.streaming)
                results.append(res)
                done += 1
                if progress is not None:
                    progress(done, len(requests))

        await asyncio.gather(*(worker() for _ in range(min(spec.concurrency, len(requests)))))
        return results

    async def _open_loop(
        self,
        client: BenchmarkClient,
        spec: BenchmarkSpec,
        requests: list[BenchmarkRequest],
        progress: ProgressCallback | None,
    ) -> list[RequestBenchmarkResult]:
        """Poisson arrivals at ``request_rate`` req/s (concurrency bounds in-flight requests)."""
        assert spec.request_rate
        rng = random.Random(spec.seed)
        sem = asyncio.Semaphore(spec.concurrency)
        results: list[RequestBenchmarkResult] = []
        done = 0

        async def one(r: BenchmarkRequest) -> None:
            nonlocal done
            async with sem:
                res = await client.run_request(r, endpoint=spec.endpoint, stream=spec.streaming)
            results.append(res)
            done += 1
            if progress is not None:
                progress(done, len(requests))

        tasks: list[asyncio.Task[None]] = []
        for r in requests:
            tasks.append(asyncio.create_task(one(r)))
            await asyncio.sleep(rng.expovariate(spec.request_rate))
        await asyncio.gather(*tasks)
        return results
