"""Controller-owned HTTP correctness tests, independent of the experimental runtime.

Without user goldens the baseline establishes regression references, not a claim of semantic
accuracy. Tests include blocking/streaming parity. Tolerances cannot be changed by proposals.
"""

from __future__ import annotations

import difflib
import math
import re
from typing import Any

import httpx

from servepilot.benchmark.client import _extract_text, _first_choice, _parse_response
from servepilot.optimization.schemas import (
    CorrectnessCase,
    CorrectnessObservation,
    CorrectnessResult,
    CorrectnessSuite,
    fingerprint,
)


def default_suite() -> CorrectnessSuite:
    prompts = [
        "What is 17 plus 25? Answer with only the number.",
        "Repeat exactly: The quick brown fox jumps over the lazy dog.",
        "Translate 'Good morning' into French. Answer briefly.",
        "List the first five prime numbers, separated by commas.",
        "Write a Python function that returns the square of its input. No explanation.",
        "In one sentence, explain why water freezes.",
    ]
    return CorrectnessSuite(
        cases=[
            CorrectnessCase(
                name=f"regression-{i + 1}",
                request={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 64,
                },
            )
            for i, prompt in enumerate(prompts)
        ]
    )


def compare(case: CorrectnessCase, expected: str, actual: str) -> bool:
    if case.comparison == "exact":
        return actual == expected
    if case.comparison == "similarity":
        return (
            difflib.SequenceMatcher(None, expected, actual, autojunk=False).ratio()
            >= case.minimum_similarity
        )
    pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    wanted = [float(n) for n in re.findall(pattern, expected)]
    got = [float(n) for n in re.findall(pattern, actual)]
    # Numeric comparison still requires identical surrounding text and the same number count.
    if (
        not wanted
        or len(wanted) != len(got)
        or re.sub(pattern, "#", expected) != re.sub(pattern, "#", actual)
    ):
        return False
    return all(
        math.isfinite(a)
        and math.isfinite(b)
        and math.isclose(a, b, rel_tol=case.relative_tolerance, abs_tol=case.absolute_tolerance)
        for a, b in zip(wanted, got, strict=True)
    )


class CorrectnessVerifier:
    def __init__(self, suite: CorrectnessSuite, *, client: httpx.AsyncClient | None = None) -> None:
        self.suite = suite.model_copy(deep=True)
        self._client = client

    async def _request(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        model: str,
        case: CorrectnessCase,
        stream: bool,
    ) -> str:
        payload: dict[str, Any] = {
            "temperature": 0,
            "seed": 42,
            "max_tokens": 64,
            **case.request,
            "model": model,
            "stream": stream,
        }
        path = "/v1/chat/completions" if case.endpoint == "chat" else "/v1/completions"
        url = base_url.rstrip("/") + path
        if not stream:
            response = await client.post(url, json=payload, timeout=self.suite.timeout_seconds)
            response.raise_for_status()
            body = _parse_response(response.content)
            if _first_choice(body).get("finish_reason") not in ("stop", "length"):
                raise ValueError("missing or unsuccessful finish_reason")
            text = _extract_text(body, case.endpoint)
        else:
            pieces = []
            finished = False
            async with client.stream(
                "POST", url, json=payload, timeout=self.suite.timeout_seconds
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        break
                    body = _parse_response(data)
                    reason = _first_choice(body).get("finish_reason")
                    if reason is not None:
                        if reason not in ("stop", "length"):
                            raise ValueError(f"unsuccessful finish_reason: {reason}")
                        finished = True
                    pieces.append(_extract_text(body, case.endpoint))
            if not finished:
                raise ValueError("incomplete stream: missing successful finish_reason")
            text = "".join(pieces)
        if not text.strip():
            raise ValueError("empty model output")
        return text

    async def run(
        self,
        base_url: str,
        model: str,
        *,
        reference: dict[str, str] | None = None,
    ) -> CorrectnessResult:
        owned = self._client is None
        client = self._client or httpx.AsyncClient()
        observations = []
        try:
            for case in self.suite.cases:
                expected = (
                    case.expected if case.expected is not None else (reference or {}).get(case.name)
                )
                # A baseline must agree with an independent repeat before becoming a reference.
                modes = [False, False] if expected is None else [False]
                if self.suite.verify_streaming:
                    modes.append(True)
                for stream in modes:
                    actual = None
                    reason = None
                    passed = False
                    try:
                        actual = await self._request(client, base_url, model, case, stream)
                        if expected is None:
                            expected = actual
                        passed = compare(case, expected, actual)
                        if not passed:
                            reason = f"{case.comparison} comparison failed"
                    except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
                        reason = f"{type(exc).__name__}: {exc}"
                    observations.append(
                        CorrectnessObservation(
                            name=case.name,
                            streaming=stream,
                            passed=passed,
                            expected=expected,
                            actual=actual,
                            reason=reason,
                        )
                    )
        finally:
            if owned:
                await client.aclose()
        return CorrectnessResult(
            suite_fingerprint=fingerprint(self.suite),
            observations=observations,
            baseline_reference=reference is None,
        )

    def references(self, result: CorrectnessResult) -> dict[str, str]:
        if not result.passed or result.suite_fingerprint != fingerprint(self.suite):
            raise ValueError("cannot use failed or incompatible correctness tests as a reference")
        return {o.name: o.expected for o in result.observations if o.expected is not None}
