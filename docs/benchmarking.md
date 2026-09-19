# Benchmarking and tuning

## Autonomous optimization

`optimize` freezes the workload, random seed, load levels, correctness cases, tolerances,
objective, and SLOs in `run.json`. It verifies streaming and blocking output before running
at least two confirmation trials per load level. All requests must succeed. Missing required
latency metrics, non-finite measurements, or violated SLOs cannot win. The score uses the
worst confirmation trial, with a default 2% improvement threshold over the incumbent.

Optimization counts generated output text with the controller's exact tokenizer, ignoring
editable engines' self-reported usage. A missing exact tokenizer stops a production run.
The general `benchmark` command and explicit fake testing mode can use approximate counts.
Raw inputs, generation settings, seeds, output text, and timings are saved alongside aggregate
measurements. YAML distributions and JSONL replay are both supported; replay preserves original
messages/prompts and generation settings. Profiler traffic is separate from scored trials.

Default correctness cases establish repeatable baseline regression references. They do not
establish general semantic accuracy. `--correctness` supplies fixed goldens with exact, numeric,
or text-similarity checks. Tolerances cannot change during a run or on resume.

The details below also describe the existing manual benchmark and staged tuner.

## Requests

Prompts are generated from a fixed word list with the model's own tokenizer, so a "512
token" prompt really has 512 tokens. The same seed always gives the same prompts. Lengths
follow the profile: 70% of requests sit within ±25% of p50, 30% spread from p50 up to p95,
and every 20th request is exactly p95. `ignore_eos` is set so the engine produces the full
requested output length (vLLM and SGLang support it; `servepilot benchmark --no-ignore-eos`
turns it off for other servers).

If no tokenizer can be loaded, an approximate word-based counter is used and the record says
so.

## Per-request measurements

| Metric | Definition |
| --- | --- |
| TTFT | time from sending the request to the first streamed chunk that carries model text. Connection setup counts; empty role-only chunks do not. |
| end-to-end latency | send to last byte |
| TPOT | `(latency − TTFT) ÷ (output tokens − 1)` |
| tokens | from the API `usage` object when present (`stream_options.include_usage`), otherwise counted with the tokenizer |

Non-streaming runs have no TTFT or TPOT: a complete response does not reveal when the first
token was generated. When `usage` is absent, output tokens are counted from the complete text,
so splitting the same output into different streaming chunks does not change the count.

Malformed responses and streamed API errors count as failed requests, even after partial
output. A stream must end with a finish reason or `[DONE]` marker to count as complete.

## Aggregates

- `duration` is the wall clock from the first request start to the last completion.
- `output_tokens_per_second` = successful output tokens ÷ duration. Never `1 ÷ mean latency`.
- `request_throughput` = successful requests ÷ duration.
- Percentiles (p50/p95/p99) are linear-interpolated over successful requests.
- `error_rate` = failed ÷ total. Failed requests count toward the duration but not the tokens.
- GPU utilization, memory and power are sampled every 0.5 s during the run when NVML is
  available. Sampling failures never fail a benchmark.

Before measuring, ServePilot sends warmup requests (`2·√concurrency`, between 2 and 16) and
requires at least one to succeed.

Closed loop is the default: `concurrency` workers each pull the next request when they
finish. `--request-rate` switches to open loop with Poisson arrivals.

## Scoring

`error_factor = max(0, 1 − 5 × error_rate)`. Results with an error rate above 1% (or no
successes) are invalid and cannot win while a valid result exists.

| Objective | Score |
| --- | --- |
| throughput | `output_tokens_per_second × error_factor` |
| latency | `(0.5·L + 0.25·T + 0.25·P) × error_factor`, where `L`, `T`, `P` are `best ÷ candidate` for p95 latency, p95 TTFT and p95 TPOT (missing metrics are dropped and the weights renormalised) |
| balanced | `sqrt(throughput_norm × latency_norm) × error_factor`, with `throughput_norm = tps ÷ best_tps` and `latency_norm = best_p95 ÷ candidate_p95` |

Exceeding a hard limit (`--max-p95-ttft`, `--max-p95-latency`, `--max-p95-tpot`) makes a result
ineligible. A missing measurement also makes a result ineligible for a limit on that metric;
for example, a non-streaming result cannot verify a TTFT limit. If nothing meets the limits,
the best valid result wins and the output says so.

## Stages

**A. Structural search.** Viable layouts in heuristic order, up to
`tuning.max_structural_candidates` (default 6), one at a time, benchmarked at half their
estimated capacity (at least 8, at most 256, at least 2 per replica), or at
`--expected-concurrency` when given. Latency runs use `--expected-concurrency` (default 16,
reported as an assumption when you did not set it).

**B. Concurrency sweep.** For the top `k` (default 2) layouts: 4, 8, 16, 32, ... up to the
layout's capacity. The sweep stops when errors appear, a limit breaks, throughput drops more
than 5% below the best point, or it plateaus (less than 2% gain while p95 latency rises more
than 10%, or under 2% gain twice in a row). Then the midpoints on both sides of the best
point are tested. With the latency objective the sweep only runs upward under a limit, to
find the highest concurrency that still meets it. If none of the top layouts can be
relaunched for its sweep, the stage A measurements decide.

**Operating point.** Walking up the sweep, a step that gains less than 2% throughput while
adding more than 10% p95 latency does not count as an improvement, so the lower concurrency
wins. This is the "plateau rule". Thresholds are constants.

**C. Memory tuning.** Raise the engine memory fraction by 0.04 (never into the last 1 GiB of
free memory). Keep it if it launches and improves the score by at least 2%. An OOM here is
recorded and the previous fraction stays. A forced `--memory-fraction` disables this stage.

**D. Confirmation.** Re-benchmark the winner at its operating point with 4× the sweep sample.
Those are the numbers that get saved and shown.

Every launch failure (OOM, crash, timeout, unsupported model) is classified from the engine
log and recorded on the candidate. The search continues with the next one.

## Isolation

Only one candidate runs at a time. After each one, ServePilot terminates the whole process
group, verifies the processes are gone, and waits for GPU memory to come back near the
baseline before launching the next. If processes cannot be terminated, tuning stops rather
than measuring on a contaminated GPU.

## What is saved

The tuning record holds the hardware snapshot, model profile, workload, seed, every benchmark
spec and result per candidate, every failure with its log tail, engine versions, the
ServePilot version, the Pareto front of (throughput, p95 latency) points, the rationale, and
the exact engine commands for the winner. `servepilot cache show KEY` prints it;
`servepilot tune -o file.json` writes it.

Cache reuse also checks the requested engine, GPU layout, context, engine arguments, KV-cache
dtype, forced memory fraction, and per-replica concurrency cap. `serve --dry-run` uses a
compatible cached result or shows a heuristic plan without launching engines.

`tune --resume` reuses completed structural measurements only when the launch configuration
and engine version still match. Changing an engine argument or memory budget remeasures the
affected layout, even if its topology ID is unchanged.
