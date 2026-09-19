"""Deterministic benchmark request generation from a :class:`WorkloadProfile`.

Prompts are synthesised from a fixed vocabulary and trimmed with the model tokenizer so their
token lengths follow the profile's p50/p95 distribution. The same seed always produces the same
requests, which keeps candidate comparisons fair and results reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from servepilot.models.tokenizer import TokenCounter
from servepilot.schemas.workload import WorkloadProfile

# A neutral vocabulary; real tokenizers split these into 1-2 tokens each, so prompts look like
# prose rather than a repeated string (which prefix caches would collapse).
_VOCABULARY = [
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "a",
    "lazy",
    "dog",
    "while",
    "seven",
    "engineers",
    "debate",
    "whether",
    "latency",
    "or",
    "throughput",
    "matters",
    "more",
    "for",
    "production",
    "systems",
    "that",
    "serve",
    "language",
    "models",
    "to",
    "thousands",
    "of",
    "users",
    "every",
    "second",
    "across",
    "regions",
    "with",
    "different",
    "network",
    "conditions",
    "and",
    "hardware",
    "budgets",
    "consider",
    "memory",
    "bandwidth",
    "compute",
    "density",
    "interconnect",
    "topology",
    "batching",
    "strategies",
    "caching",
    "scheduling",
    "fairness",
    "cost",
    "energy",
    "reliability",
    "observability",
    "and",
    "the",
    "many",
    "trade",
    "offs",
    "involved",
    "when",
    "planning",
    "capacity",
    "for",
    "unpredictable",
    "demand",
    "patterns",
    "during",
    "peak",
    "hours",
    "weekends",
    "holidays",
    "product",
    "launches",
    "and",
    "incidents",
    "that",
    "require",
    "rapid",
    "mitigation",
    "careful",
    "analysis",
    "and",
    "clear",
    "communication",
    "between",
    "teams",
    "responsible",
    "for",
    "infrastructure",
    "applications",
    "and",
    "customer",
    "success",
]


@dataclass(frozen=True)
class BenchmarkRequest:
    index: int
    prompt: str
    input_tokens: int
    max_tokens: int
    shared_prefix: str = ""
    payload: dict[str, Any] | None = None
    endpoint: str | None = None


class PromptGenerator:
    def __init__(self, tokenizer: TokenCounter, seed: int) -> None:
        self._tok = tokenizer
        self._seed = seed
        self._rng = random.Random(seed)
        # Tokens per vocabulary word for this tokenizer, measured once on a fixed sample.
        sample = " ".join(_VOCABULARY)
        self._tokens_per_word = max(0.25, self._tok.count(sample) / len(_VOCABULARY))

    def _word_list(self, rng: random.Random, n: int) -> list[str]:
        return [rng.choice(_VOCABULARY) for _ in range(n)]

    def text_with_tokens(self, target_tokens: int, rng: random.Random) -> tuple[str, int]:
        """Produce text whose token count is as close as possible to ``target_tokens``.

        The word count is predicted from the measured tokens-per-word ratio and corrected in a
        few rounds; exact tokenizers then trim to the precise token count via decode/encode.
        """
        target = max(1, target_tokens)
        words = self._word_list(rng, max(1, round(target / self._tokens_per_word)))
        ids = self._tok.encode(" ".join(words))
        for _ in range(4):
            deficit = target - len(ids)
            if abs(deficit) <= max(1, target // 100):
                break
            if deficit > 0:
                words.extend(self._word_list(rng, max(1, round(deficit / self._tokens_per_word))))
            else:
                drop = max(1, round(-deficit / self._tokens_per_word))
                words = words[: max(1, len(words) - drop)]
            ids = self._tok.encode(" ".join(words))
        text = " ".join(words)
        if len(ids) > target and self._tok.exact:
            text = self._tok.decode(ids[:target]).strip()
            ids = self._tok.encode(text)
        return text, len(ids)

    @staticmethod
    def sample_length(rng: random.Random, p50: int, p95: int) -> int:
        """Sample a length whose distribution has roughly the requested p50 and p95.

        70% of samples are drawn around p50 (±25%), 30% span the p50→p95 range so the tail is
        exercised; the p95 value itself is guaranteed to appear in every batch of 20.
        """
        if p95 <= p50:
            return p50
        if rng.random() < 0.7:
            return max(1, int(p50 * rng.uniform(0.75, 1.25)))
        return int(rng.uniform(p50, p95))

    def generate(
        self, workload: WorkloadProfile, num_requests: int, *, seed_offset: int = 0
    ) -> list[BenchmarkRequest]:
        rng = random.Random(self._seed + seed_offset)
        shared_prefix = ""
        if workload.shared_prefix_fraction > 0:
            prefix_tokens = int(workload.input_tokens_p50 * workload.shared_prefix_fraction)
            shared_prefix, _ = self.text_with_tokens(prefix_tokens, random.Random(self._seed))
        requests: list[BenchmarkRequest] = []
        for i in range(num_requests):
            # Every 20th request is a p95 request so tails are always represented.
            if i % 20 == 19:
                in_tokens = workload.input_tokens_p95
                out_tokens = workload.output_tokens_p95
            else:
                in_tokens = self.sample_length(
                    rng, workload.input_tokens_p50, workload.input_tokens_p95
                )
                out_tokens = self.sample_length(
                    rng, workload.output_tokens_p50, workload.output_tokens_p95
                )
            body_tokens = max(
                1, in_tokens - (self._tok.count(shared_prefix) if shared_prefix else 0)
            )
            text, measured = self.text_with_tokens(body_tokens, rng)
            prompt = f"{shared_prefix} {text}".strip() if shared_prefix else text
            total_in = measured + (self._tok.count(shared_prefix) if shared_prefix else 0)
            requests.append(
                BenchmarkRequest(
                    index=i,
                    prompt=prompt,
                    input_tokens=total_in,
                    max_tokens=max(1, min(out_tokens, workload.max_context_tokens - total_in)),
                    shared_prefix=shared_prefix,
                )
            )
        return requests


def warmup_count(concurrency: int, minimum: int, maximum: int) -> int:
    """Warmup requests grow modestly with concurrency (≈ sqrt) within bounds."""
    import math

    return max(minimum, min(maximum, math.ceil(math.sqrt(concurrency) * 2)))
