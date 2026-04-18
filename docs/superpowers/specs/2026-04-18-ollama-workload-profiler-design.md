# Ollama Workload Profiler V1 Design

## Objective

Build a publishable GitHub project named `ollama-workload-profiler`: a CLI-first local benchmarking and profiling tool for Ollama that helps a user understand what a selected local model is good for on their machine, which use cases deserve caution, how context sizing changes usability, where CPU and RAM usage peak, how cold and warm runs differ, and how interactive the model feels through TTFT where selected.

V1 is intentionally bounded:

- One model per benchmark session
- One or more user-selected contexts per session
- One or more user-selected benchmark families per session
- Deterministic, source-controlled prompts and scenarios
- Windows first, Linux supported where practical
- Local-only Ollama API integration and local-only telemetry

## Product Boundaries

### In Scope

- Python CLI tool with installable app command
- Standalone bootstrap flow
- Interactive CLI session setup
- Local Ollama API integration
- Raw metrics capture and local process/system sampling
- Markdown and machine-readable reporting
- Deterministic verdict engine
- Pinned dependencies and reproducible setup

### Out Of Scope For V1

- GUI
- Cloud APIs
- Multi-model sessions
- Non-Ollama backends
- Distributed serving
- Perfect hardware telemetry beyond practical local metrics
- Silent or brittle Ollama auto-installation

## Command Model

V1 uses a split command model.

### Standalone Bootstrap

Command:

```powershell
python scripts/bootstrap.py
```

Responsibilities:

- Detect Python environment state
- Detect whether execution is already inside a venv
- Create or validate a venv
- Install or repair pinned Python dependencies
- Check Ollama binary installation
- Check Ollama reachability
- Check whether any local models exist
- Print explicit remediation guidance

Bootstrap rules:

- Safe to rerun
- Idempotent
- Must not run benchmarks
- Must not silently mutate unrelated system state
- May automate venv creation and Python dependency installation only
- Must not auto-install Ollama in V1 unless a future flow is truly robust

### Installable App CLI

Commands:

- `owp doctor`
- `owp profile`

`owp doctor` is read-only and reports:

- OS and platform details
- Python version
- Venv status
- Dependency status
- Ollama binary detected or not
- Ollama reachable or not
- Local models found or not
- Likely remediation hints

It exits non-zero on critical failures.

`owp doctor` is only expected to run after bootstrap or package installation has completed. It is an installed app command, not a replacement for the standalone bootstrap flow.

`owp profile` runs the interactive benchmark flow, refuses to start when critical prerequisites fail unless a future safe override exists, and writes one session artifact set for one selected model.

## Runtime Model

The core runtime contract is a typed `BenchmarkSessionPlan`. It is a first-class object, not scattered CLI state.

`BenchmarkSessionPlan` owns:

- Exactly one selected model
- Ordered list of selected context sizes
- Ordered list of selected benchmark types
- One repetition count
- Optional stop conditions
- Session timestamps
- Execution settings

The user-selected session intent is persisted as `plan.json`.

Before execution begins, the planner expands the session plan into a flat, fully ordered run list and persists it as `expanded_plan.json`.

This distinction is locked:

- `plan.json`: user-selected intent
- `expanded_plan.json`: concrete ordered execution list

The benchmark report is one report per session and one model per session. Multiple selected contexts are compared inside that single report.

## Session Timing

Two timers exist:

- Bootstrap global timer starts immediately when `scripts/bootstrap.py` starts
- Benchmark session timer starts immediately when `owp profile` starts

Bootstrap time is never mixed into the final benchmark session duration reported by `owp profile`.

Persist machine-friendly UTC timestamps in artifacts and render user-friendly local timestamps in terminal output and Markdown reporting.

## Execution Ordering

Execution is deterministic and context-first.

For each selected context, the run order is:

1. `smoke`
2. `cold-warm`
3. `prompt-scaling`
4. `context-scaling`
5. `output-scaling`
6. `use-case-profiles`
7. `ttft`
8. `stress`

Scenario and repetition ordering are deterministic within each benchmark family.

Each concrete run carries:

- `run_index`: monotonic integer in execution order
- `run_id`: unique run identifier
- `context_index`
- `benchmark_type_index`
- `scenario_index`
- `repetition_index`

This metadata is persisted in raw results so reporting never depends on filename order.

## Module Layout

V1 package layout:

- `src/ollama_workload_profiler/cli.py`
- `src/ollama_workload_profiler/session.py`
- `src/ollama_workload_profiler/env_check.py`
- `src/ollama_workload_profiler/ollama_client.py`
- `src/ollama_workload_profiler/benchmarks/`
- `src/ollama_workload_profiler/metrics/`
- `src/ollama_workload_profiler/prompts/`
- `src/ollama_workload_profiler/models/`
- `src/ollama_workload_profiler/reporting/`
- `scripts/bootstrap.py`
- `tests/`

Recommended internal benchmark modules:

- `benchmarks/base.py` or `benchmarks/runner.py`
- `benchmarks/smoke.py`
- `benchmarks/cold_warm.py`
- `benchmarks/prompt_scaling.py`
- `benchmarks/context_scaling.py`
- `benchmarks/output_scaling.py`
- `benchmarks/profiles.py`
- `benchmarks/ttft.py`
- `benchmarks/stress.py`

Recommended model modules:

- `models/plan.py`
- `models/results.py`
- `models/verdicts.py`
- `models/failures.py`
- `models/summary.py`

## Typed Schemas

Persisted models must remain JSON-compatible. Do not store Python-only runtime objects inside persisted artifacts.

### `BenchmarkSessionPlan`

Represents session intent:

- selected model
- ordered contexts
- ordered benchmark types
- repetition count
- stop conditions
- timestamps
- execution settings

Serializable and persisted as `plan.json`.

### `ScenarioDefinition`

Represents a deterministic benchmark scenario definition.

Required fields:

- `scenario_id`
- `benchmark_type`
- `name`
- `version`
- `prompt_payload`
- `target_output_tokens`

Optional fields:

- `profile_tag`
- `fill_ratio`
- `difficulty_tag`
- `phase_emphasis`

Locked prompt rules:

- deterministic
- source-controlled
- inspectable
- versioned

No hidden content generation at runtime beyond deterministic template assembly.

### `RunResult`

One record per executed run. It is append-only in spirit and should be finalized once from execution facts, then treated as immutable by reporting code.

Fields include:

- selected model
- selected context
- benchmark type
- scenario name and `scenario_id`
- repetition index
- ordering metadata
- wall-clock elapsed time
- Ollama timing metrics
- local resource aggregates
- phase peaks
- stop state
- failure state
- partial-result indicators
- artifact references

Rule:

- `RunResult` describes what happened in one run
- `ReportSummary` describes what the session means

### `VerdictInput`

Normalized, intentionally smaller than `RunResult`. Contains only fields needed by the verdict engine, such as:

- TTFT band
- generation speed band
- RAM headroom or pressure band
- success or failure state
- scenario type
- context size

### `ReportSummary`

Report-ready aggregate state:

- executive summary
- context comparisons
- suitability matrix
- warnings and failures
- recommendations
- references to artifact outputs

### `FailureInfo`

Explicit failure schema used by runs and summaries. Includes:

- failure kind
- lifecycle phase
- message
- exception class if applicable
- threshold exceeded if applicable
- stop-trigger metadata if applicable

### `PhasePeak`

Explicit phase-level resource schema. Supported coarse phases:

- `load`
- `prompt_ingestion`
- `generation`
- `resident_post_run`

Use enums for benchmark type, run phase, verdict label, failure kind, and scenario family to reduce stringly-typed bugs.

## Environment Snapshot

Persist a session-wide environment snapshot as `environment.json`.

Include:

- OS and platform
- Python version
- package version
- Ollama version if detectable
- model metadata snapshot from Ollama
- benchmark start timestamps

Keep collection practical and non-invasive. Avoid overcollecting host details.

## Planner And Run Lifecycle

The planner takes a `BenchmarkSessionPlan`, expands benchmark families into concrete `ScenarioDefinition` entries, produces a fully ordered list of executable runs, and persists that order before execution begins.

Run lifecycle:

1. `planned`
2. `starting`
3. `load`
4. `prompt_ingestion`
5. `generation`
6. `resident_post_run`
7. terminal state:
   - `completed`
   - `stopped`
   - `failed`

A run is considered started immediately before request dispatch, at the same moment resource sampling begins. That boundary is used consistently for logs, TTFT, run metadata, and failure timing.

Default consecutive execution policy:

- Continue to the next run after a per-run failure unless a stop condition or fatal session error requires ending the session

Fatal session errors are intentionally small in scope:

- Ollama becomes unreachable for the remainder of the session
- Selected model no longer exists
- Artifact directory cannot be written
- Sampler infrastructure cannot start at all when required metrics depend on it

## Sampling And Metrics Flow

Per run:

1. capture plan and ordering metadata
2. create run record in `planned`
3. transition to `starting`
4. start sampler
5. dispatch Ollama request
6. collect request and stream events
7. collect Ollama metrics
8. stop sampler after resident snapshot
9. aggregate local metrics
10. derive phase peaks
11. classify terminal state
12. finalize immutable `RunResult`
13. persist run result

Sampling window:

- starts just before request dispatch
- stops after the post-run resident-state snapshot

Add a small configurable settling window for the `resident_post_run` phase to reduce instantaneous noise.

Phase attribution rules:

- deterministic
- coarse
- based on known timestamps and metric windows
- never pseudo-precise beyond what Ollama and local timing support

### Required Ollama Metrics

Collect when available:

- `total_duration`
- `load_duration`
- `prompt_eval_count`
- `prompt_eval_duration`
- `eval_count`
- `eval_duration`

Derived metrics may include:

- prompt tokens per second
- generation tokens per second
- effective total tokens per second

### Required Local Metrics

Collect via local-only sampling:

- peak Ollama process RAM
- average Ollama process RAM
- peak Ollama process CPU
- average Ollama process CPU
- peak system RAM used

Optional if practical:

- process thread count
- process I/O counters
- paging or swap indicators

### Wall-Clock Fallback

Always record whole-run local wall-clock elapsed time even when Ollama timing fields are present. This acts as:

- a sanity check
- a fallback when Ollama omits or malforms metrics
- a measure of actual time the user waited

### Telemetry Failure Policy

Telemetry failure is not always run failure.

Rules:

- If required metrics are missing, the run is `failed`
- If optional metrics are missing, the run may still be `completed` or `stopped` with warnings

Required normal-run facts:

- model
- context
- benchmark type
- `scenario_id`
- run ordering metadata
- final state
- total duration from Ollama or local wall-clock fallback
- core success or failure metadata

Required TTFT facts:

- request start timestamp
- first token timestamp, or explicit absence that leads to stop or failure classification
- final state

Partial results are still persisted when failures occur mid-run. Partiality must be explicit in the raw result schema and reporting.

### Metrics Classification Table

| Metric class | Required fields |
| --- | --- |
| Normal run required | model, context, benchmark type, `scenario_id`, run ordering metadata, final state, total duration from Ollama or local wall-clock fallback, core success or failure metadata |
| TTFT run required | request start timestamp, first token timestamp or explicit absence that drives stop or failure classification, final state |
| Optional with warning | Ollama sub-metrics that are absent but not required for run classification, local optional telemetry such as I/O counters, thread count, paging or swap indicators, and any non-critical telemetry that fails while core run facts remain intact |

## Session Continuation Policy

Session continuation is explicit and policy-driven.

| Event type | Default behavior |
| --- | --- |
| Per-run failure | Continue to the next run |
| Stop condition | Continue or skip later runs according to the configured stop rule |
| Fatal session error | Abort the session |

This section governs session flow independently from individual run terminal state classification.

## Stop, Failure, And Success Semantics

Use these terminal states exactly:

- `completed`
- `stopped`
- `failed`

### `completed`

Use only when:

- Ollama returned a usable final response
- required metrics for that run were captured
- artifact writing did not invalidate the run result

### `stopped`

Use only for intentional, policy-driven halts:

- threshold stop
- skip rule
- operator-configured stop condition

### `failed`

Use for actual execution faults:

- request or transport error
- malformed Ollama response
- unexpected timeout
- unrecoverable telemetry failure
- internal exception

This distinction must be explicit in schema and reporting language.

## Stop Conditions

V1 supports optional stop conditions such as:

- stop if RAM exceeds threshold
- stop if TTFT exceeds threshold
- stop if generation tokens per second drops below threshold
- skip larger contexts if a smaller context already failed badly

Stopped runs are marked as stopped, never silently treated as passed.

## Benchmark Family Catalog

Each benchmark family expands into a small bounded number of scenarios per selected context. V1 must not allow combinatorial explosion.

| Benchmark family | Scenarios per context | Purpose |
| --- | ---: | --- |
| `smoke` | 1 | Basic sanity check |
| `cold-warm` | 2 | Compare cold-ish and warm behavior |
| `prompt-scaling` | 3 | Isolate prompt-ingestion cost |
| `context-scaling` | 3 | Pressure-test fill ratio within selected context |
| `output-scaling` | 3 | Isolate generation-length behavior |
| `use-case-profiles` | 6 | Evaluate practical task suitability |
| `ttft` | 2 | Measure first-token responsiveness with streaming |
| `stress` | 2 | Bounded stress testing with stop conditions |

Total:

- 22 scenarios per selected context before repetition

### `smoke`

- 1 scenario per context
- short deterministic prompt
- short output target

### `cold-warm`

- 2 scenarios per context
- same prompt and same output target
- first run attempts cold behavior where feasible
- second run is immediate warm repeat

Cold behavior must degrade gracefully. Persist:

- requested cold run
- actual cold-prep method used

Do not claim a run was truly cold when only a best-effort cooldown or unload approach was possible.

Cold-run preparation uses this ordered fallback policy:

1. Try explicit unload or equivalent safe model-eviction behavior if Ollama supports it
2. Else wait a configured cooldown period
3. Else mark the run as best-effort cold

The raw result must persist both the requested cold behavior and the actual preparation method used.

### `prompt-scaling`

- 3 scenarios per context
- same task type
- varying prompt sizes
- fixed short output target

Default target bands:

- about `256`
- about `1024`
- about `4096`

### `context-scaling`

- 3 scenarios per selected context
- low, medium, high fill ratios
- synthetic fill relative to the already selected context
- fixed short output target

This family does not change the selected session context. It pressure-tests usage within that selected context.

Default fill ratio targets:

- about `25%`
- about `50%`
- about `80%`

### `output-scaling`

- 3 scenarios per context
- same prompt
- short, medium, long output targets

Default targets:

- about `64`
- about `256`
- about `768`

### `use-case-profiles`

- 6 scenarios per context
- one deterministic scenario each for:
  - `quick_qa`
  - `long_document_summary`
  - `code_explanation`
  - `structured_extraction`
  - `long_form_generation`
  - `multi_turn_chat`

Additional rules:

- `multi_turn_chat` uses a fixed source-controlled turn sequence
- long-document summary uses static fixture text
- code explanation uses static fixture code

### `ttft`

- exactly 2 scenarios per context
- streaming required
- optimized for first-token measurement rather than throughput

### `stress`

- exactly 2 scenarios per context
- bounded
- stop-rule compatible
- never open-ended

Each stress scenario defines:

- max runtime
- max output target
- stop-rule compatibility

## Pre-Run Budget Note

Before execution starts, `owp profile` should display a compact budget summary that includes:

- total concrete run count
- total scenario count before repetition expansion
- repetition count
- a rough warning when the selected plan is likely to be large for slower hardware

V1 does not need precise duration prediction, but it should help users notice obviously large benchmark matrices before confirming execution.

## Prompt And Token Count Policy

All prompts and scenario fixtures must be deterministic, source-controlled, versioned, clearly named, easy to inspect, and easy to modify.

Each scenario may additionally carry:

- `difficulty_tag`: `light`, `medium`, `heavy`
- `phase_emphasis`: `load_sensitive`, `prompt_sensitive`, `generation_sensitive`, `ttft_sensitive`, `stress_sensitive`

Token policy:

- scenario definitions store target bands
- actual token counts vary by model
- Ollama-reported prompt counts are the source of truth

Persist both:

- target band metadata
- actual observed token counts from Ollama

## Reporting

Each session writes to a unique timestamped results directory. No prior session is overwritten.

Required artifacts:

- `plan.json`
- `expanded_plan.json`
- `environment.json`
- `raw.jsonl`
- `raw.csv`
- `summary.json`
- `report.md`

Artifact filenames are fixed in V1 to simplify automation and testing.

Report sections:

1. Executive summary
2. Per-model summary
3. Use-case suitability matrix
4. Detailed timing and resource tables
5. Phase-level resource peaks
6. Warnings and failures
7. Plain-language recommendations

The report must contain both:

- raw numbers
- deterministic verdict labels

Allowed verdict labels:

- Good fit
- Good fit with caution
- Use with caution
- Use only for short tasks
- Avoid on this hardware

Verdict logic lives in a dedicated module and must be deterministic, readable, and easy to tune.

Wall-clock and Ollama timing precedence:

- persist both wall-clock and Ollama timing data
- use wall-clock duration for "time the user waited"
- use Ollama timing fields for internal phase analysis when present

## Use-Case Suitability Matrix

The V1 use-case suitability matrix is explicitly shaped as:

- rows: use-case profiles
- columns: selected contexts
- cells: verdict label plus a compact supporting metric tuple

Do not mix general benchmark families into this matrix in V1.

## Prompt Fixture Versioning

Scenario and prompt fixture version information must be persisted into raw results, not only kept in source definitions, so later reruns remain comparable even if fixtures evolve.

## Testing Strategy

At minimum, V1 includes tests for:

- environment detection
- Ollama discovery logic
- interactive selection validation helpers
- verdict engine
- report rendering
- benchmark plan generation

Core logic must be testable without a live Ollama instance. Ollama-dependent tests should rely on mocks, and test runs must not require local model downloads.

## Implementation Notes

- Prefer robust behavior over cleverness
- Prefer explicit user messaging over silent magic
- Keep modules small and composable
- Separate benchmark execution, metrics collection, and reporting
- Isolate OS-specific process discovery and telemetry behind clean interfaces
- Do not overbuild V1
