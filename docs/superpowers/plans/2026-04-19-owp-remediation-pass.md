# Ollama Workload Profiler Remediation Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the current `main` branch from audit status `not ready` to a release-shape V1 by correcting execution ordering, interactive profile UX, persisted artifact semantics, telemetry, TTFT, and reporting without rewriting the project.

**Architecture:** This pass keeps the existing module layout and corrects contracts in place. `expanded_plan.json` becomes the single source of truth for execution order, profile startup becomes interactive and session-oriented, run persistence becomes append-per-run, and reporting consumes richer raw run facts instead of inferring state late.

**Tech Stack:** Python 3.11+, Typer, Rich, httpx, psutil, pydantic, Jinja2, pytest, pytest-mock

---

## File Structure

Files to modify during remediation:

- Modify: `src/ollama_workload_profiler/cli.py`
- Modify: `src/ollama_workload_profiler/session.py`
- Modify: `src/ollama_workload_profiler/ollama_client.py`
- Modify: `src/ollama_workload_profiler/models/plan.py`
- Modify: `src/ollama_workload_profiler/models/results.py`
- Modify: `src/ollama_workload_profiler/models/failures.py`
- Modify: `src/ollama_workload_profiler/reporting/artifacts.py`
- Modify: `src/ollama_workload_profiler/reporting/summary.py`
- Modify: `src/ollama_workload_profiler/reporting/markdown.py`
- Modify: `src/ollama_workload_profiler/reporting/verdicts.py`
- Modify: `src/ollama_workload_profiler/metrics/process.py`
- Modify: `src/ollama_workload_profiler/metrics/sampler.py`
- Modify: `src/ollama_workload_profiler/metrics/phases.py`
- Modify: `src/ollama_workload_profiler/benchmarks/ttft.py`
- Modify: `src/ollama_workload_profiler/benchmarks/stress.py`
- Modify: `src/ollama_workload_profiler/benchmarks/output_scaling.py`
- Modify: `src/ollama_workload_profiler/benchmarks/cold_warm.py`
- Modify: `README.md`
- Modify: `tests/test_cli_validation.py`
- Modify: `tests/test_session.py`
- Modify: `tests/test_reporting.py`
- Modify: `tests/test_verdicts.py`
- Modify: `tests/test_ollama_client.py`

## Task 1: Make Expanded Plan The Sole Execution Contract

**Files:**
- Modify: `src/ollama_workload_profiler/models/plan.py`
- Modify: `src/ollama_workload_profiler/models/results.py`
- Modify: `src/ollama_workload_profiler/session.py`
- Test: `tests/test_session.py`

**Acceptance criteria:**
- `expand_session_plan()` emits runs in spec order: contexts first, then fixed benchmark family order.
- `expanded_plan.json` order matches the exact execution order.
- No later layer reorders runs after expansion.
- Raw run results carry ordering metadata required by reporting.

## Task 2: Restore Interactive Profile Session Startup

**Files:**
- Modify: `src/ollama_workload_profiler/cli.py`
- Modify: `src/ollama_workload_profiler/session.py`
- Test: `tests/test_cli_validation.py`

**Acceptance criteria:**
- `owp profile` can run as an interactive flow instead of requiring all flags.
- The flow checks prerequisites early, shows a budget summary, and requires confirmation before execution.
- The selected model, contexts, and benchmark types produce one concrete expanded plan before any run starts.

## Task 3: Persist Session Artifacts Incrementally

**Files:**
- Modify: `src/ollama_workload_profiler/reporting/artifacts.py`
- Modify: `src/ollama_workload_profiler/session.py`
- Test: `tests/test_reporting.py`
- Test: `tests/test_cli_validation.py`

**Acceptance criteria:**
- Session directory is created at session start.
- `plan.json` and `expanded_plan.json` are written before the first run.
- `raw.jsonl` and `raw.csv` are appended per run rather than written only at the end.
- Session timestamps reflect real session start time.

## Task 4: Replace Placeholder Telemetry With Real Sampling

**Files:**
- Modify: `src/ollama_workload_profiler/metrics/process.py`
- Modify: `src/ollama_workload_profiler/metrics/sampler.py`
- Modify: `src/ollama_workload_profiler/metrics/phases.py`
- Modify: `src/ollama_workload_profiler/session.py`
- Modify: `src/ollama_workload_profiler/models/results.py`
- Test: `tests/test_session.py`

**Acceptance criteria:**
- Profile runs use a non-noop sampler.
- Core process sampling starts immediately before dispatch and stops after post-run sampling.
- Phase peaks are persisted into raw run results.
- Required telemetry failures produce the intended run classification.

## Task 5: Implement Real TTFT Semantics

**Files:**
- Modify: `src/ollama_workload_profiler/ollama_client.py`
- Modify: `src/ollama_workload_profiler/session.py`
- Modify: `src/ollama_workload_profiler/benchmarks/ttft.py`
- Modify: `src/ollama_workload_profiler/models/results.py`
- Test: `tests/test_ollama_client.py`
- Test: `tests/test_session.py`

**Acceptance criteria:**
- TTFT scenarios use streaming.
- `ttft_ms` is derived from request-start to first-token arrival.
- TTFT raw results persist request start and first token facts or explicit absence when classification depends on it.

## Task 6: Correct Reporting And Release-Facing Outputs

**Files:**
- Modify: `src/ollama_workload_profiler/reporting/summary.py`
- Modify: `src/ollama_workload_profiler/reporting/markdown.py`
- Modify: `src/ollama_workload_profiler/reporting/verdicts.py`
- Modify: `README.md`
- Test: `tests/test_reporting.py`
- Test: `tests/test_verdicts.py`

**Acceptance criteria:**
- Summary metrics derive from expanded plan and raw facts, not shortcuts.
- Use-case matrix cells render readable verdict labels and compact metrics, not Python repr strings.
- Failure summaries use the correct lifecycle semantics.
- README no longer documents known release-blocking drift as current behavior.

## Task 7: Close Remaining Benchmark Catalog Drift

**Files:**
- Modify: `src/ollama_workload_profiler/benchmarks/ttft.py`
- Modify: `src/ollama_workload_profiler/benchmarks/stress.py`
- Modify: `src/ollama_workload_profiler/benchmarks/output_scaling.py`
- Modify: `src/ollama_workload_profiler/benchmarks/cold_warm.py`
- Test: `tests/test_session.py`

**Acceptance criteria:**
- TTFT family emits 2 scenarios per context.
- Stress family emits 2 bounded scenarios per context.
- Output-scaling targets match the intended short/medium/long bands.
- Cold-warm runs persist requested cold behavior and actual prep method.

## Verification Notes

- Run focused tests after each task slice before moving on.
- Run the full suite before claiming the remediation pass is complete.
- Keep `main` clean between task slices except for the intended changes.
