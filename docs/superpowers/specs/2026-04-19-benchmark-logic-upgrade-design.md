# Benchmark Logic Upgrade Design

## Objective

Upgrade `ollama-workload-profiler` from a useful local profiler into a credible blog-grade local benchmark tool for Ollama runs on a single PC.

This pass stays intentionally local-first and practical:

- Improve measurement honesty, repeatability, and interpretability
- Preserve the current architecture, with `session.py` as the orchestration center
- Avoid broader model-quality evaluation scope
- Avoid a structural refactor unless required for a P0 fix

## Scope

### In Scope For This Pass

- Deterministic generation settings exposed in the CLI and persisted in artifacts
- Real cold versus warm execution control
- Per-context warmup for non-cold scenarios
- Explicit load-time, prompt-time, and generation-time metrics
- Token-targeted context fill with bounded calibration and caching
- Expanded environment and model metadata suitable for blog reporting
- Aggregate summaries centered on median and p95 using completed runs only
- README updates that explain benchmark semantics honestly

### Out Of Scope For This Pass

- Standardized model-quality benchmarks such as MMLU, GSM8K, or lm-eval-harness
- Distributed or remote benchmarking
- Heavy concurrency or load-testing infrastructure
- Dashboarding or visualization work beyond current reports
- A large refactor of the runtime into new orchestration layers

## Design Principles

- Labels must match enforced behavior. If the tool says "cold", the code must perform explicit cold-state preparation.
- Approximate measurements must be labeled as approximate.
- Benchmark settings must be first-class artifact data, not hidden inside unrelated metadata blobs.
- Summary numbers should discourage cherry-picking by default.
- Bounded extra work is acceptable for better honesty; unbounded calibration loops are not.

## Architecture

`session.py` remains the control plane for execution planning, per-run preparation, warmup policy, metric extraction, and artifact-facing execution metadata.

Supporting modules are extended rather than replaced:

- `cli.py` adds benchmark controls and passes them into the session plan
- `ollama_client.py` grows small API helpers for unload, preload, version, and model metadata
- `prompts/scenarios.py` continues to define scenarios, but context-fill scenarios become calibration-aware instead of char-count based
- `env_check.py` becomes the home for richer environment and Ollama metadata detection
- `reporting/summary.py` and markdown rendering consume the richer metrics and present less flattering but more trustworthy aggregates

The existing append-only artifact flow remains intact.

## Artifact Model

Benchmark settings become a distinct first-class artifact contract.

### `plan.json`

`BenchmarkSessionPlan.execution_settings` becomes the durable source of requested execution policy, including:

- `seed`
- `temperature`
- `top_p`
- `repetitions`
- `warmup_runs`
- `warmup_enabled`
- calibration bounds and tolerance if exposed or internally fixed

This is the user's requested benchmark policy.

### `environment.json`

`environment.json` continues to capture host and runtime metadata, but now also includes:

- Ollama version
- detected accelerator information
- model metadata returned by Ollama where available
- benchmark settings copied in a read-friendly summary section for report consumers

### Per-run artifacts

Each run row in `raw.jsonl` and `raw.csv` records:

- requested prep behavior
- actual prep method used
- whether prep enforcement succeeded
- calibration target, actual prompt tokens, and calibration status when applicable
- deterministic generation settings used for the request

If cold-state enforcement or calibration fails, the run is still recorded, but the artifacts must say so explicitly.

## Execution Settings

The CLI exposes:

- `--seed` with default `42`
- `--temperature` with default `0.0`
- `--top-p` optional and unset by default
- `--repetitions`
- `--warmup-runs`
- `--no-warmup`

These settings feed `BenchmarkSessionPlan.execution_settings` and are echoed in the benchmark budget output so the user sees the benchmark policy before execution starts.

`_OllamaDispatcher._build_options()` uses these settings for every measured request, with `tokens_per_second` retained only as a backward-compatible alias of generation TPS.

## Cold, Warm, And Warmup Semantics

Cold and warm semantics are enforced in the session orchestrator rather than implied by scenario labels alone.

### Cold scenarios

For a cold scenario:

1. Explicitly unload the model before the measured request using the Ollama API path that best maps to `keep_alive=0`
2. Record whether unload was requested and whether the call succeeded
3. Skip generic warmup for that measured run
4. Run the measured request

If unload cannot be enforced, the run remains labeled with the requested behavior but records that cold-state preparation was not enforced successfully.

### Warm scenarios

For a warm scenario:

1. Issue a short preload request for the same model and context
2. Keep the model resident
3. Run the measured request
4. Record that warm preparation was explicitly performed

### General session warmup

For non-cold benchmark families, the session performs silent warmup per context size by default.

Warmup policy:

- Warmup is tracked per selected context size, not once per session
- Default is one short silent warmup request per context size before the first measured run at that context
- `--warmup-runs` controls how many silent warmup runs happen
- `--no-warmup` disables this behavior entirely
- Cold scenarios bypass this default warmup path so they remain truly cold attempts

## Metric Model

Per-run response metrics expand to include:

- `load_duration`
- `load_duration_ms`
- `prompt_eval_count`
- `prompt_eval_duration`
- `eval_count`
- `eval_duration`
- `prompt_tokens_per_second`
- `generation_tokens_per_second`
- `tokens_per_second` as an alias of `generation_tokens_per_second`

This preserves backward compatibility while making prompt-side and generation-side behavior explicit.

TTFT remains supported in its current form for P0. A split between first emission and first answer token is not required for this pass and should only be included if it emerges as a low-risk addition inside already touched code.

## Context Fill Calibration

The current char-length approximation is replaced by bounded token-targeted calibration.

### Scenario intent

Context scenarios continue to declare:

- requested fill ratio
- target output tokens

They also gain calibration-facing metadata such as:

- target token count derived from `context_size * fill_ratio`
- tolerance band
- calibration status

### Calibration loop

For context-fill scenarios, the session orchestrator:

1. Builds a candidate prompt from stable source text
2. Sends a bounded calibration request
3. Reads actual `prompt_eval_count`
4. Expands or shrinks the prompt
5. Stops when actual prompt tokens land inside tolerance or when the max calibration attempts are exhausted

Calibration must be cached. The cache key should include at least:

- model name
- context size
- scenario id or fill ratio
- generation settings that materially affect tokenization if any

The cache may live in memory for the session; persistent cross-session caching is not required for this pass.

### Bounded behavior

Calibration must not run indefinitely. The design uses:

- a fixed maximum attempt count
- a tolerance such as plus or minus five percent
- explicit fallback recording when calibration does not converge

If calibration does not converge, the scenario still runs, but the artifacts and reports must say that the fill is approximate and include requested versus actual prompt token counts.

## Environment And Model Metadata

The environment snapshot grows to answer the questions a blog reader will immediately ask.

### Host metadata

- CPU basics already captured remain
- GPU detection adds practical best-effort detection:
  - NVIDIA via `nvidia-smi`
  - AMD via `rocm-smi` when available
  - Apple Silicon detection on macOS
  - otherwise an explicit "not detected" or likely CPU-only outcome

### Ollama metadata

- Ollama version
- `/api/show` model metadata where available
- quantization level if exposed
- model size or parameter size if exposed
- model context length if exposed
- processor split or related residency hints if practical through Ollama inspection

Metadata collection is best-effort and non-fatal. Missing commands or unavailable Ollama fields are recorded as unavailable, not fabricated.

## Aggregate Reporting

Aggregate summaries should reflect completed runs only and include sample size.

### Session and benchmark summaries

Replace or demote flattering defaults such as average elapsed and max TPS. Headline benchmark stats should include:

- `sample_size`
- `completed_runs`
- `failed_runs`
- `stopped_runs`
- `median_elapsed_ms`
- `p95_elapsed_ms`
- `median_prompt_tokens_per_second`
- `p95_prompt_tokens_per_second`
- `median_generation_tokens_per_second`
- `p95_generation_tokens_per_second`
- `median_ttft_ms` where applicable
- `p95_ttft_ms` where applicable
- `median_load_duration_ms`
- `p95_load_duration_ms`

Mean values may remain only as secondary context if already useful, but they should not be the lead benchmark framing.

### Markdown report wording

The report should:

- explain that this tool measures local serving and workload behavior
- explain how cold and warm were enforced
- flag approximate context-fill results when calibration did not converge
- use unambiguous metric names for prompt TPS and generation TPS

## Error Handling And Honesty Rules

This pass prefers explicit honesty over silent fallback.

- If cold unload fails, record `requested_prep_behavior="cold_start"` and an enforcement status that says cold prep was not achieved.
- If warm preload fails, record the failure explicitly.
- If calibration fails to converge, record the run as approximate with requested and observed token counts.
- If metadata cannot be detected, record it as unavailable instead of omitting context silently.

Execution should continue where possible so the session still produces inspectable artifacts.

## Testing Strategy

The implementation follows TDD and extends existing tests in place.

Coverage focus:

- CLI parsing and budget echo for new benchmark settings
- dispatcher request options for deterministic generation
- cold and warm orchestration behavior in `run_profile_session`
- per-context warmup behavior and cold bypass logic
- response metric extraction for load, prompt TPS, and generation TPS
- calibration cache and bounded retry behavior
- explicit artifact persistence for benchmark settings and prep/calibration status
- summary calculations using completed runs only with sample size, median, and p95
- metadata helpers for Ollama and accelerator detection with best-effort failure handling

## Implementation Boundaries

This design intentionally avoids:

- introducing a new orchestration subsystem
- changing benchmark family scope beyond what P0 requires
- adding cross-session calibration storage
- redefining TTFT semantics unless a low-risk addition naturally falls out of touched code

The goal is a clean, credible P0 upgrade inside the current architecture.
