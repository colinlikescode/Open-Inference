"""Real CPU inference processes exercise optimization, rollback, resume, and deployment."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from servepilot.models.inspector import ModelInspector
from servepilot.models.tokenizer import ApproximateTokenizer
from servepilot.optimization.agent import AgentTool, AgentTurn, PiAgent
from servepilot.optimization.budget import TimeBudget
from servepilot.optimization.report import load_recipe
from servepilot.optimization.store import ExperimentStore
from servepilot.optimization.workflow import OptimizeOptions, deploy_recipe, optimize
from servepilot.runtime.ports import ephemeral_port
from servepilot.runtime.state import RuntimeStateStore
from servepilot.schemas.model import ModelProfile
from servepilot.settings import ServePilotSettings


@pytest.fixture
def simulation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dense_8b: ModelProfile
) -> tuple[OptimizeOptions, ServePilotSettings]:
    monkeypatch.setattr(ModelInspector, "inspect", lambda *_args, **_kwargs: dense_8b)
    monkeypatch.setattr(
        "servepilot.optimization.workflow.load_tokenizer",
        lambda *_args, **_kwargs: ApproximateTokenizer(),
    )
    monkeypatch.setattr(PiAgent, "preflight", lambda _: None)
    traffic = tmp_path / "traffic.yaml"
    traffic.write_text(
        yaml.safe_dump(
            {
                "workload": {
                    "input_tokens": {"p50": 8, "p95": 12},
                    "output_tokens": {"p50": 4, "p95": 8},
                    "concurrency": {"expected": 1, "peak": 2},
                }
            }
        )
    )
    correctness = tmp_path / "correctness.yaml"
    correctness.write_text(
        yaml.safe_dump(
            {
                "cases": [
                    {
                        "name": "stable",
                        "endpoint": "completions",
                        "request": {"prompt": "Complete this", "max_tokens": 4},
                    }
                ]
            }
        )
    )
    behavior = tmp_path / "behavior.json"
    behavior.write_text(
        json.dumps(
            {
                "default": {"startup_delay_s": 0, "ttft_ms": 10, "tpot_ms": 15},
                "plans": {
                    "faster": {"startup_delay_s": 0, "ttft_ms": 1, "tpot_ms": 1},
                    "crashed": {"startup_delay_s": 0, "startup": "crash"},
                },
            }
        )
    )
    settings = ServePilotSettings(
        state_dir=tmp_path / "state",
        cache_dir=tmp_path / "cache",
        enable_fake_engine=True,
        fake_hardware="h100x1",
        fake_engine_behavior=behavior,
    )
    return OptimizeOptions(
        model="test/model",
        engine="fake",
        seconds=60,
        output=tmp_path / "output",
        workload=traffic,
        correctness=correctness,
        concurrency=[1, 2],
        requests=4,
        deploy=False,
    ), settings


async def test_optimize_accepts_improvement_rejects_crash_and_resumes_without_rewriting(
    simulation: tuple[OptimizeOptions, ServePilotSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    options, settings = simulation
    calls = 0

    async def turn(
        self: PiAgent, context: dict[str, Any], tools: list[AgentTool], budget: TimeBudget
    ) -> AgentTurn:
        nonlocal calls
        calls += 1
        if calls <= 2:
            plan = context["current_best"]["proposal"]["plan"].copy()
            plan["id"] = "faster" if calls == 1 else "crashed"
            submit = next(t for t in tools if t.name == "submit_experiment")
            await submit.handler({"hypothesis": plan["id"], "plan": plan})
        return AgentTurn(text="complete", events=[], tool_calls=int(calls <= 2))

    monkeypatch.setattr(PiAgent, "turn", turn)
    messages = []
    outcome = await optimize(options, settings, messages.append)
    assert outcome.best is not None, messages
    assert outcome.best.proposal.plan.id == "faster"
    store = ExperimentStore(options.output)
    results = store.results()
    assert [r.status for r in results] == ["verified", "verified", "failed"]
    assert results[1].decision.accepted
    assert not results[2].decision.accepted
    assert all(r.correctness and r.correctness.passed for r in results[:2])
    assert len(list((options.output / "benchmark-results").glob("*-requests.json"))) == 8
    old_events = (options.output / "experiments.jsonl").read_bytes()
    old_results = [(store.experiment_dir(r.id) / "result.json").read_bytes() for r in results]
    saved = load_recipe(options.output / "recipe.yaml")
    assert saved.experiment == 2 and saved.result.runtime["testing_only"]
    resumed = await optimize(
        OptimizeOptions(resume=options.output, seconds=30, deploy=False), settings, messages.append
    )
    assert resumed.best and resumed.best.id == 2
    assert (options.output / "experiments.jsonl").read_bytes().startswith(old_events)
    assert old_results == [
        (store.experiment_dir(r.id) / "result.json").read_bytes() for r in results
    ]
    assert not (settings.state_dir / "optimization.json").exists()


async def test_saved_recipe_launches_clean_server_without_agent(
    simulation: tuple[OptimizeOptions, ServePilotSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    options, settings = simulation

    async def no_proposal(*_args: Any) -> AgentTurn:
        return AgentTurn(text="done", events=[], tool_calls=0)

    monkeypatch.setattr(PiAgent, "turn", no_proposal)
    outcome = await optimize(options, settings, lambda _: None)
    assert outcome.best
    monkeypatch.setattr(
        PiAgent, "preflight", lambda _: pytest.fail("recipe deployment must not invoke Pi")
    )
    port = ephemeral_port()
    ready = asyncio.Event()
    task = asyncio.create_task(
        deploy_recipe(
            options.output / "recipe.yaml",
            nodes=None,
            settings=settings,
            host="127.0.0.1",
            port=port,
            bootstrap=False,
            progress=lambda message: ready.set() if message.startswith("Ready:") else None,
        )
    )
    try:
        async with asyncio.timeout(20):
            while not ready.is_set():
                if task.done():
                    await task
                    pytest.fail("deployment exited before readiness")
                await asyncio.sleep(0.05)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/v1/completions",
                json={"model": "llama3_8b", "prompt": "hello", "max_tokens": 4},
            )
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["text"] == "tok0 tok1 tok2 tok3"
        state = RuntimeStateStore(settings.state_dir).read()
        assert state and state.children
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert RuntimeStateStore(settings.state_dir).read() is None


async def test_no_slo_compliant_candidate_produces_report_without_recipe(
    simulation: tuple[OptimizeOptions, ServePilotSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    options, settings = simulation
    options.max_p95_ttft_ms = 0.001

    async def no_proposal(*_args: Any) -> AgentTurn:
        return AgentTurn(text="done", events=[], tool_calls=0)

    monkeypatch.setattr(PiAgent, "turn", no_proposal)
    outcome = await optimize(options, settings, lambda _: None)
    assert outcome.best is None
    assert (options.output / "report.html").is_file()
    assert not (options.output / "recipe.yaml").exists()


async def test_deadline_kills_hung_startup_and_resume_retries_baseline(
    simulation: tuple[OptimizeOptions, ServePilotSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    from servepilot.engines.process import LocalLauncher

    options, settings = simulation
    assert settings.fake_engine_behavior
    settings.fake_engine_behavior.write_text(
        json.dumps({"default": {"startup": "hang", "startup_delay_s": 0}})
    )
    options.seconds = 1
    launchers = []

    class RecordingLauncher(LocalLauncher):
        def __init__(self) -> None:
            super().__init__()
            launchers.append(self)

    monkeypatch.setattr("servepilot.optimization.workflow.LocalLauncher", RecordingLauncher)

    async def no_proposal(*_args: Any) -> AgentTurn:
        return AgentTurn(text="done", events=[], tool_calls=0)

    monkeypatch.setattr(PiAgent, "turn", no_proposal)
    outcome = await optimize(options, settings, lambda _: None)
    assert outcome.best is None and outcome.stopping_reason == "budget_expired"
    results = ExperimentStore(options.output).results()
    assert results and results[0].status == "interrupted"
    assert all(not launcher.tracked() for launcher in launchers)
    settings.fake_engine_behavior.write_text(
        json.dumps({"default": {"startup_delay_s": 0, "ttft_ms": 1, "tpot_ms": 1}})
    )
    resumed = await optimize(
        OptimizeOptions(resume=options.output, seconds=30, deploy=False), settings, lambda _: None
    )
    assert resumed.best and resumed.best.id == 2
    assert ExperimentStore(options.output).results()[0].status == "interrupted"


async def test_setup_deadline_saves_checkpoint_and_resumes(
    simulation: tuple[OptimizeOptions, ServePilotSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from servepilot.testing.fake_hardware import FakeHardwareProvider

    options, settings = simulation
    snapshot = FakeHardwareProvider.snapshot

    def slow_snapshot(provider: FakeHardwareProvider):
        time.sleep(0.1)
        return snapshot(provider)

    monkeypatch.setattr(FakeHardwareProvider, "snapshot", slow_snapshot)
    options.seconds = 0.02
    outcome = await optimize(options, settings, lambda _: None)
    assert outcome.stopping_reason == "setup_budget_expired"
    assert (options.output / ".setup.json").is_file()
    assert (options.output / "report.html").is_file()
    assert not (options.output / "run.json").exists()
    monkeypatch.setattr(FakeHardwareProvider, "snapshot", snapshot)

    async def no_proposal(*_args: Any) -> AgentTurn:
        return AgentTurn(text="done", events=[], tool_calls=0)

    monkeypatch.setattr(PiAgent, "turn", no_proposal)
    resumed = await optimize(
        OptimizeOptions(resume=options.output, seconds=30, deploy=False), settings, lambda _: None
    )
    assert resumed.best and resumed.best.id == 1
    assert load_recipe(options.output / "recipe.yaml").definition.model.model_id == "llama3_8b"
