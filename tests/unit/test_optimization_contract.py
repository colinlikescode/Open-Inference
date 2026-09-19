"""Acceptance and history invariants: an agent cannot turn bad evidence into a win."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from servepilot.benchmark.client import build_payload
from servepilot.benchmark.runner import make_spec
from servepilot.exceptions import ConfigurationError
from servepilot.models.tokenizer import ApproximateTokenizer
from servepilot.optimization.budget import BudgetExpired, TimeBudget
from servepilot.optimization.correctness import CorrectnessVerifier, compare
from servepilot.optimization.report import file_sha256, load_recipe, write_recipe
from servepilot.optimization.schemas import (
    AgentConfig,
    CorrectnessCase,
    CorrectnessObservation,
    CorrectnessResult,
    CorrectnessSuite,
    EvaluationPolicy,
    ExperimentProposal,
    ExperimentResult,
    RunDefinition,
    RuntimeFile,
    VerificationDecision,
    fingerprint,
    utc_now,
)
from servepilot.optimization.store import ExperimentStore
from servepilot.optimization.verifier import DeterministicVerifier
from servepilot.optimization.workload import load_traffic, replay_requests
from servepilot.schemas.benchmark import BenchmarkResult
from servepilot.schemas.hardware import HardwareSnapshot
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import CandidatePlan, EngineName
from servepilot.schemas.workload import LatencyConstraints, WorkloadProfile


@pytest.fixture
def definition(dense_8b: ModelProfile, h100x1: HardwareSnapshot) -> RunDefinition:
    return RunDefinition(
        model=dense_8b,
        hardware=h100x1,
        workload=WorkloadProfile(),
        correctness=CorrectnessSuite(
            cases=[
                CorrectnessCase(
                    name="sum",
                    request={"prompt": "2+2="},
                    endpoint="completions",
                    expected="4",
                )
            ]
        ),
        agent=AgentConfig(model="inference-engineer"),
        policy=EvaluationPolicy(concurrency_levels=[2], requests_per_trial=4),
    )


def evidence(definition: RunDefinition) -> CorrectnessResult:
    return CorrectnessResult(
        suite_fingerprint=fingerprint(definition.correctness),
        observations=[
            CorrectnessObservation(name="sum", streaming=s, passed=True, expected="4", actual="4")
            for s in (False, True)
        ],
    )


def measurements(definition: RunDefinition, throughput: float = 100) -> list[BenchmarkResult]:
    spec = make_spec(
        definition.workload, concurrency=2, num_requests=4, seed=definition.policy.seed
    )
    result = BenchmarkResult(
        candidate_id="test",
        spec=spec,
        total_requests=4,
        successful_requests=4,
        failed_requests=0,
        duration_seconds=1,
        request_throughput=4,
        input_tokens_per_second=20,
        output_tokens_per_second=throughput,
        total_tokens_per_second=throughput + 20,
        ttft_p95_ms=10,
        tpot_p95_ms=5,
        latency_p50_ms=30,
        latency_p95_ms=40,
        latency_p99_ms=50,
        error_rate=0,
    )
    return [result.model_copy(deep=True), result.model_copy(deep=True)]


def proposal() -> ExperimentProposal:
    return ExperimentProposal(
        hypothesis="baseline",
        plan=CandidatePlan(
            id="test",
            engine=EngineName.FAKE,
            gpu_groups=[[0]],
            tensor_parallel_size=1,
            replica_count=1,
            context_length=8192,
        ),
    )


def experiment(definition: RunDefinition) -> ExperimentResult:
    return ExperimentResult(
        id=1,
        proposal=proposal(),
        status="verified",
        started_at=utc_now(),
        elapsed_seconds=1,
        correctness=evidence(definition),
        benchmarks=measurements(definition),
        decision=VerificationDecision(
            eligible=True, accepted=True, score=100, recommended_concurrency=2
        ),
    )


@pytest.mark.parametrize("mutation", ["corrupt", "missing", "symlink_escape"])
def test_recipe_rejects_changed_runtime_archives(
    tmp_path: Path, definition: RunDefinition, mutation: str
) -> None:
    store = ExperimentStore(tmp_path / "output")
    with store:
        store.create(definition)
    archive = store.directory / "images" / "engine.tar"
    archive.parent.mkdir()
    archive.write_bytes(b"recorded engine image bytes")
    result = experiment(definition)
    result.artifacts["images/engine.tar"] = file_sha256(archive)
    result.runtime["image_archives"] = {"local": "images/engine.tar"}
    path = write_recipe(store, definition, result)
    assert load_recipe(path).files["images/engine.tar"] == file_sha256(archive)

    if mutation == "corrupt":
        archive.write_bytes(b"different engine image bytes")
    else:
        original = archive.read_bytes()
        archive.unlink()
        if mutation == "symlink_escape":
            outside = tmp_path / "outside.tar"
            outside.write_bytes(original)
            archive.symlink_to(outside)
    with pytest.raises(ConfigurationError, match=r"checksum mismatch|missing artifact or escaping"):
        load_recipe(path)


def test_confirmation_run_must_improve(definition: RunDefinition) -> None:
    verifier = DeterministicVerifier(definition)
    results = measurements(definition, 200)
    results[1].output_tokens_per_second = 101
    decision = verifier.decide(evidence(definition), results, incumbent=experiment(definition))
    assert decision.eligible and not decision.accepted
    results[1].output_tokens_per_second = 110
    assert verifier.decide(evidence(definition), results, incumbent=experiment(definition)).accepted


@pytest.mark.parametrize("mutation", ["failed", "missing", "nan", "seed", "samples", "streaming"])
def test_bad_benchmark_evidence_fails_closed(definition: RunDefinition, mutation: str) -> None:
    results = measurements(definition, 10000)
    if mutation == "failed":
        results[0].failed_requests = 1
    elif mutation == "missing":
        results.pop()
    elif mutation == "nan":
        results[0].output_tokens_per_second = float("nan")
    elif mutation == "seed":
        results[0].spec.seed += 1
    elif mutation == "samples":
        results[0].total_requests -= 1
    else:
        results[0].spec.streaming = False
    result = DeterministicVerifier(definition).decide(evidence(definition), results)
    assert not result.accepted and not result.eligible


def test_slo_failure_or_missing_metric_never_falls_back(definition: RunDefinition) -> None:
    definition.workload.latency_constraints = LatencyConstraints(max_p95_ttft_ms=5)
    verifier = DeterministicVerifier(definition)
    results = measurements(definition)
    assert not verifier.decide(evidence(definition), results).accepted
    results[0].ttft_p95_ms = None
    assert not verifier.decide(evidence(definition), results).eligible


def test_no_correctness_or_missing_case_cannot_win(definition: RunDefinition) -> None:
    verifier = DeterministicVerifier(definition)
    assert not verifier.decide(None, measurements(definition)).eligible
    partial = evidence(definition)
    partial.observations.pop()
    assert not verifier.decide(partial, measurements(definition)).eligible


def test_agent_cannot_submit_a_score() -> None:
    payload = proposal().model_dump()
    payload["score"] = 999999
    with pytest.raises(ValidationError, match="Extra inputs"):
        ExperimentProposal.model_validate(payload)


def test_verifier_contract_isolated_and_checked(definition: RunDefinition) -> None:
    verifier = DeterministicVerifier(definition)
    definition.policy.minimum_improvement = 0.5
    assert verifier.definition.policy.minimum_improvement == 0.02
    verifier.definition.policy.minimum_improvement = 0
    with pytest.raises(RuntimeError, match="mutated"):
        verifier.decide(None, [])


def test_history_cannot_overwrite_or_silently_resume_tampering(
    tmp_path: Path, definition: RunDefinition
) -> None:
    with ExperimentStore(tmp_path / "run") as store:
        store.create(definition)
        assert store.begin(proposal()) == 1
        store.finish(experiment(definition))
        with pytest.raises(ConfigurationError, match="immutable"):
            store.finish(experiment(definition))
    with ExperimentStore(tmp_path / "run") as store:
        assert len(store.results()) == 1
        assert fingerprint(store.definition()) == fingerprint(definition)
        assert not store.pending()
        path = store.experiment_dir(1) / "result.json"
        data = json.loads(path.read_text())
        data["decision"]["score"] = 100000
        path.write_text(json.dumps(data))
        with pytest.raises(ConfigurationError, match="checksum"):
            store.results()


def test_journal_chain_detects_modified_history(tmp_path: Path, definition: RunDefinition) -> None:
    with ExperimentStore(tmp_path) as store:
        store.create(definition)
        store.begin(proposal())
        path = tmp_path / "experiments.jsonl"
        path.write_text(path.read_text().replace("baseline", "cheating"))
        with pytest.raises(ConfigurationError, match="checksum"):
            store.events()


def test_output_lock_and_artifact_escape(tmp_path: Path, definition: RunDefinition) -> None:
    with ExperimentStore(tmp_path) as store:
        store.create(definition)
        with (
            pytest.raises(ConfigurationError, match="another controller"),
            ExperimentStore(tmp_path),
        ):
            pass
        with pytest.raises(ConfigurationError, match="escapes"):
            store.artifact("../escape", b"x")
        store.artifact("kernels/test.py", b"first")
        with pytest.raises(ConfigurationError, match="historical"):
            store.artifact("kernels/test.py", b"changed")


@pytest.mark.parametrize(
    "path", ["../run.json", "/etc/passwd", "foo/../../escape", ".agent/key", "a\\b"]
)
def test_runtime_files_are_confined(path: str) -> None:
    with pytest.raises(ValidationError):
        RuntimeFile(path=path, content="anything")


async def test_budget_cancels_running_work_and_allows_cleanup() -> None:
    cleaned = asyncio.Event()

    async def operation() -> None:
        try:
            await asyncio.sleep(10)
        finally:
            cleaned.set()

    budget = TimeBudget(0.02)
    with pytest.raises(BudgetExpired):
        await budget.run(operation)
    assert cleaned.is_set()
    with pytest.raises(BudgetExpired):
        await budget.run(operation)


async def test_correctness_detects_wrong_answer_and_incomplete_stream() -> None:
    suite = CorrectnessSuite(
        cases=[
            CorrectnessCase(
                name="sum",
                request={"prompt": "2+2="},
                endpoint="completions",
                expected="4",
            )
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["stream"]:
            return httpx.Response(200, text='data: {"choices":[{"text":"4"}]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, json={"choices": [{"text": "5", "finish_reason": "stop"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await CorrectnessVerifier(suite, client=client).run("http://backend", "model")
    assert not result.passed
    assert "comparison failed" in (result.observations[0].reason or "")
    assert "incomplete stream" in (result.observations[1].reason or "")


def test_numeric_tolerance_does_not_ignore_non_numeric_changes() -> None:
    case = CorrectnessCase(
        name="value",
        endpoint="completions",
        request={"prompt": "x"},
        comparison="numeric",
        absolute_tolerance=0.01,
    )
    assert compare(case, "value=1.0", "value=1.001")
    assert not compare(case, "value=1.0", "wrong=1.0")
    assert not compare(case, "value=1.0", "value=9")


def test_actual_jsonl_messages_and_generation_are_replayed(tmp_path: Path) -> None:
    path = tmp_path / "requests.jsonl"
    payload = {
        "messages": [
            {"role": "system", "content": "Be brief"},
            {"role": "user", "content": "Real traffic"},
        ],
        "max_tokens": 50,
        "temperature": 0.3,
    }
    path.write_text(json.dumps(payload) + "\n")
    _, _, entries = load_traffic(path, WorkloadProfile(), ApproximateTokenizer())
    request = replay_requests(entries, ApproximateTokenizer())[0]
    sent = build_payload(request, model="target", endpoint="chat", stream=True)
    assert sent["messages"] == payload["messages"]
    assert sent["temperature"] == 0.3 and sent["max_tokens"] == 50
    assert "ignore_eos" not in sent


def test_yaml_workload_preserves_long_tail_and_peak(tmp_path: Path) -> None:
    path = tmp_path / "workload.yaml"
    path.write_text("""workload:
  input_tokens: {p50: 2000, p95: 12000}
  output_tokens: {p50: 500, p95: 2000}
  concurrency: {expected: 64, peak: 128}
""")
    profile, levels, replay = load_traffic(path, WorkloadProfile(), ApproximateTokenizer())
    assert profile.max_context_tokens >= 14000
    assert levels == [32, 64, 128] and not replay
