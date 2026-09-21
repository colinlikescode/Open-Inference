"""Human-readable capacity report and a verified, integrity-checked deployment recipe."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any

import yaml

from servepilot.exceptions import ConfigurationError
from servepilot.optimization.loop import OptimizationOutcome
from servepilot.optimization.schemas import (
    DeploymentRecipe,
    ExperimentResult,
    RunDefinition,
    fingerprint,
)
from servepilot.optimization.store import ExperimentStore, atomic_write


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:,.2f}"


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capacity_chart(result: ExperimentResult | None) -> str:
    if result is None or not result.benchmarks:
        return ""
    levels = sorted({measurement.concurrency for measurement in result.benchmarks})
    points = [
        (
            level,
            min(m.output_tokens_per_second for m in result.benchmarks if m.concurrency == level),
        )
        for level in levels
    ]
    maximum = max(value for _, value in points)
    if maximum <= 0:
        return ""
    marks = []
    coords = []
    for index, (level, value) in enumerate(points):
        x = 75 + index * 470 / max(1, len(points) - 1)
        y = 185 - value / maximum * 150
        coords.append(f"{x:.1f},{y:.1f}")
        marks.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#276c9b"><title>Concurrency {level}: {value:.1f} output tok/s</title></circle><text x="{x:.1f}" y="205" text-anchor="middle">{level}</text><text x="{x:.1f}" y="{y - 12:.1f}" text-anchor="middle">{value:,.0f}</text>'
        )
    return f'<figure><svg viewBox="0 0 620 240" role="img" aria-label="Measured capacity curve" style="max-width:720px;width:100%;font:12px system-ui"><path d="M55 25 V185 H575" fill="none" stroke="#8594a3"/><polyline points="{" ".join(coords)}" fill="none" stroke="#276c9b" stroke-dasharray="4 4"/>{"".join(marks)}<text x="60" y="15">Output tokens / second</text><text x="310" y="235" text-anchor="middle">Configured concurrency</text></svg><figcaption>Lowest throughput across confirmation trials at each tested load. Connecting lines do not represent additional measurements.</figcaption></figure>'


def write_report(
    store: ExperimentStore,
    definition: RunDefinition,
    outcome: OptimizationOutcome,
    *,
    endpoint: str | None = None,
) -> Path:
    results = store.results()
    winner = outcome.best
    events = store.events()
    timings: dict[str, Any] = {
        kind: next((event["payload"] for event in reversed(events) if event["kind"] == kind), {})
        for kind in (
            "search_started",
            "setup_complete",
            "artifacts_saved",
            "deployment_ready",
            "deployment_failed",
            "deployment_stopped",
        )
    }
    simulation = "fake" in definition.engine_versions
    capacity = []
    if winner is not None:
        for measurement in winner.benchmarks:
            capacity.append(
                f"<tr><td>{measurement.concurrency}</td><td>{_number(measurement.output_tokens_per_second)}</td>"
                f"<td>{_number(measurement.ttft_p95_ms)}</td><td>{_number(measurement.tpot_p95_ms)}</td>"
                f"<td>{_number(measurement.latency_p95_ms)}</td><td>{measurement.failed_requests}</td></tr>"
            )
    history = []
    for result in results:
        verdict = "accepted" if result.decision.accepted else result.status
        history.append(
            f"<tr><td>{result.id}</td><td>{_escape(result.proposal.hypothesis)}</td>"
            f"<td>{_escape(result.proposal.plan.label())}</td><td>{verdict}</td>"
            f"<td>{_escape(result.error or '; '.join(result.decision.reasons))}</td>"
            f'<td><a href="experiments/experiment_{result.id:04d}/result.json">evidence</a></td></tr>'
        )
    baseline = outcome.baseline

    def summary(result: ExperimentResult | None) -> str:
        if result is None or not result.benchmarks:
            return "No complete benchmark"
        concurrency = result.decision.recommended_concurrency or result.benchmarks[0].concurrency
        measured = [r for r in result.benchmarks if r.concurrency == concurrency]
        return (
            f"{min(r.output_tokens_per_second for r in measured):,.0f} output tok/s; "
            f"p95 TTFT {_number(max((r.ttft_p95_ms for r in measured if r.ttft_p95_ms is not None), default=None))} ms; "
            f"p95 TPOT {_number(max((r.tpot_p95_ms for r in measured if r.tpot_p95_ms is not None), default=None))} ms; "
            f"concurrency {concurrency}"
        )

    improvement = None
    if baseline and winner and baseline.decision.score and winner.decision.score:
        improvement = winner.decision.score / baseline.decision.score - 1
    page = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Open Sandbox optimization report</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1200px;margin:3rem auto;padding:0 1.5rem;color:#17212d;background:#f9fafb}}h1,h2{{color:#132f4c}}table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem;background:white}}th,td{{border:1px solid #d9e1e8;padding:.65rem;text-align:left;vertical-align:top}}th{{background:#e9eef4}}code{{overflow-wrap:anywhere}}.notice{{padding:1rem;border-left:4px solid #276c9b;background:#edf5fa}}</style>
<h1>Open Sandbox optimization report</h1>
{'<p class="notice"><strong>CPU SIMULATION:</strong> fake inference and hardware. These numbers are not GPU performance measurements.</p>' if simulation else ""}
<p class="notice">Best verified configuration discovered within the allocated experiment budget. This is not a claim of a global optimum.</p>
<p><strong>Model:</strong> {_escape(definition.model.model_id)} · <strong>Revision:</strong> {_escape(definition.model.revision or "local model")}<br>
<strong>Cluster:</strong> {len(definition.hardware.nodes) or 1} GPU node(s), {definition.hardware.gpu_count} GPUs<br>
<strong>Objective:</strong> {_escape(definition.workload.objective.value)} · <strong>Search elapsed:</strong> {outcome.elapsed_seconds:.1f}s · <strong>Stopping reason:</strong> {_escape(outcome.stopping_reason)}<br>
<strong>Endpoint:</strong> {_escape(endpoint or "not deployed")}<br>
<strong>Baseline:</strong> {_escape(summary(baseline))}<br>
<strong>Best:</strong> {_escape(summary(winner))}<br>
<strong>Objective improvement:</strong> {f"{improvement:+.1%}" if improvement is not None else "unavailable"}<br>
<strong>Recommended production concurrency:</strong> {_escape(winner.decision.recommended_concurrency if winner else "none")}<br>
<strong>Highest tested concurrency satisfying constraints:</strong> {_escape(winner.decision.maximum_slo_concurrency if winner else "none")}</p>
<h2>Verification scope</h2>
<p>{len(definition.correctness.cases)} fixed correctness cases; streaming checks {"enabled" if definition.correctness.verify_streaming else "disabled"}. All required checks must pass. Cases without user-supplied goldens compare against repeatable baseline outputs; these are regression checks, not a general proof of model quality.</p>
<p>Every load level uses {definition.policy.repetitions} confirmation trials. Selection uses the worst score across trials, rejects failed requests, and enforces the configured SLOs. Untested load levels and different hardware have no implied performance guarantee.</p>
<pre>{_escape(json.dumps({"latency_constraints": definition.workload.model_dump(mode="json")["latency_constraints"], "minimum_output_tokens_per_second": definition.policy.minimum_output_tokens_per_second}, indent=2))}</pre>
<h2>Capacity measurements for the selected deployment</h2>
{capacity_chart(winner)}
<p>Each row is a measured confirmation trial; no capacity is extrapolated beyond the tested loads.</p>
<table><thead><tr><th>Concurrency</th><th>Output tok/s</th><th>P95 TTFT ms</th><th>P95 TPOT ms</th><th>P95 latency ms</th><th>Failed requests</th></tr></thead><tbody>{"".join(capacity)}</tbody></table>
<h2>Run timing and deployment</h2>
<pre>{_escape(json.dumps(timings, indent=2))}</pre>
<p>The search budget includes inspection, setup, and verification. Final cleanup, exact image export, and deployment may add wall-clock time.</p>
<h2>Complete experiment history</h2>
<table><thead><tr><th>ID</th><th>Hypothesis</th><th>Layout</th><th>Decision</th><th>Reason</th><th>Details</th></tr></thead><tbody>{"".join(history)}</tbody></table>
<h2>Reproduction</h2><p><a href="recipe.yaml">Deployment recipe</a> · <a href="experiments.jsonl">Event history</a> · <a href="run.json">Immutable run definition</a></p>
<pre>{_escape(json.dumps({"engine_versions": definition.engine_versions, "engine_images": definition.engine_images}, indent=2))}</pre>
</html>
"""
    path = store.directory / "report.html"
    atomic_write(path, page.encode())
    return path


def write_recipe(
    store: ExperimentStore, definition: RunDefinition, result: ExperimentResult
) -> Path:
    # Preserve the evaluated limits; production concurrency is selected by the verifier.
    files = dict(result.artifacts)
    runtime = result.runtime
    for item in runtime.get("artifact_paths", []):
        path = store.directory / item
        if not path.resolve().is_relative_to(store.directory) or not path.is_file():
            raise ConfigurationError(f"selected runtime artifact is missing or invalid: {item}")
        files[item] = file_sha256(path)
    dockerfile = runtime.get("dockerfile")
    if dockerfile:
        atomic_write(store.directory / "Dockerfile", str(dockerfile).encode())
        files["Dockerfile"] = hashlib.sha256(str(dockerfile).encode()).hexdigest()
    launch = '#!/bin/sh\nset -eu\nexec servepilot deploy "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/recipe.yaml" "$@"\n'
    atomic_write(store.directory / "launch.sh", launch.encode(), mode=0o700)
    files["launch.sh"] = hashlib.sha256(launch.encode()).hexdigest()
    recipe = DeploymentRecipe(
        run_fingerprint=fingerprint(definition),
        experiment=result.id,
        definition=definition,
        result=result,
        files=files,
    )
    path = store.directory / "recipe.yaml"
    atomic_write(path, yaml.safe_dump(recipe.model_dump(mode="json"), sort_keys=False).encode())
    return path


def load_recipe(path: Path) -> DeploymentRecipe:
    try:
        recipe = DeploymentRecipe.model_validate(yaml.safe_load(path.read_text()))
        root = path.parent.resolve()
        for relative, expected in recipe.files.items():
            artifact = (root / relative).resolve()
            if not artifact.is_relative_to(root) or not artifact.is_file():
                raise ValueError(f"missing artifact or escaping path: {relative}")
            if file_sha256(artifact) != expected:
                raise ValueError(f"artifact checksum mismatch: {relative}")
        return recipe
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"invalid deployment recipe {path}: {exc}") from exc
