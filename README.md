# Ollama Workload Profiler

`ollama-workload-profiler` is a CLI-first local benchmarking and profiling tool for Ollama on a single machine.

It is designed for local serving and workload comparisons, not scientific benchmarking or model-quality evaluation. It does not run standardized quality suites such as MMLU or GSM8K. The goal is to make local Ollama runs more repeatable, more honest, and easier to explain in a blog post or hardware write-up.

## Install

The repo ships with pinned dependencies in both `requirements.txt` and `pyproject.toml`.
The bootstrap script creates or reuses `.venv`, installs dependencies, installs the project itself into that venv with an editable install, and prints a plain status line for each step.

```powershell
python scripts/bootstrap.py
```

Because the project is installed in editable mode, local changes under `src/` are reflected immediately without setting a custom `PYTHONPATH`.

## Usage

Bootstrap first, then either activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
python -m ollama_workload_profiler --help
python -m ollama_workload_profiler doctor
owp doctor
```

You can also skip activation and call the venv interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m ollama_workload_profiler doctor
.\.venv\Scripts\owp.exe doctor
```

Run help first if you are new to the CLI, then start with a small profile:

```powershell
owp --help
owp profile --model llama3.2 --contexts 4096,8192 --benchmark-types smoke,use-case-profiles
```

`owp doctor` reports platform, Python, venv status, dependency status, Ollama binary detection, Ollama reachability, local model availability, and remediation hints when something is missing.

`owp profile` supports an interactive flow by default: it can prompt for model, contexts, and benchmark families, shows a benchmark budget summary, and asks for confirmation before execution. You can still pass `--model`, `--contexts`, and `--benchmark-types` directly when scripting.

The current CLI accepts `smoke`, `cold-warm`, `prompt-scaling`, `context-scaling`, `output-scaling`, `use-case-profiles`, `ttft`, `stress`, and `concurrency-smoke`.

Results are written to timestamped directories under `results/` by default.

## Quick recipes

Run a quick smoke plus TTFT session:

```powershell
owp profile --model llama3.2 --contexts 4096 --benchmark-types smoke,ttft --yes
```

Run a small same-machine contention smoke session:

```powershell
owp profile --model llama3.2 --contexts 4096 --benchmark-types concurrency-smoke --yes
```

Export lightweight SVG plots from a completed session:

```powershell
owp export-plots results/session-2026-04-19_T13-22-05Z
```

Compare two completed sessions:

```powershell
owp compare results/session-a results/session-b
owp compare results/session-a results/session-b --strict --all-metrics
```

Compare without rerunning benchmarks and write JSON output:

```powershell
owp compare results/session-2026-04-20_T10-00-00Z results/session-2026-04-21_T10-00-00Z
owp compare results/session-a results/session-b --format json --output compare.json
```

The comparison direction is `baseline -> candidate`; deltas are candidate minus baseline. `--strict` exits nonzero for strict-blocking comparability issues. `owp compare` is read-only. It reports deltas from existing `summary.json`, `plan.json`, and `environment.json` artifacts, warns when sessions are not directly comparable under the same benchmark contract, and does not make statistical significance claims. Warnings are comparability/context signals, not proof of a meaningful performance difference.

## Benchmark semantics

The profiler treats benchmark policy as first-class session data. Requested execution settings such as `seed`, `temperature`, `top_p`, `repetitions`, and warmup behavior are written into session artifacts so runs are easier to reproduce and explain later.

Session artifacts also include `benchmark_methodology_version`, currently `bmk-v2`, in `plan.json`, `environment.json`, and `summary.json`. This is the benchmark measurement-contract version, separate from the package or CLI version. Maintainers should bump it when metric definitions, preparation/enforcement rules, aggregation eligibility, calibration semantics, or artifact interpretation change; ordinary CLI UX, documentation, formatting, or bugfix releases that preserve the measurement contract should only bump the tool version.

Cold and warm runs are enforced, not just labeled:

- `cold-warm` cold scenarios explicitly unload the model before the measured run
- `cold-warm` warm scenarios explicitly preload the model before the measured run
- non-cold scenarios use a bounded session warmup once per `(model, context_size)` boundary unless warmup is disabled

If a requested cold or warm preparation step cannot be enforced, the run artifacts say so directly and the run is excluded from strict aggregates.

## Context-fill calibration

`context-scaling` scenarios are token-targeted rather than character-targeted.

For each context-fill scenario, the profiler:

1. chooses a target prompt token count from the requested fill ratio
2. probes Ollama and reads the returned `prompt_eval_count`
3. adjusts the prompt and retries up to a bounded limit

Calibration is cached per `(model, context_size, scenario_id, prompt_template_version)` so repetitions reuse the measured prompt instead of recalibrating every run.

Calibration results are recorded as:

- `exact`: actual prompt tokens landed within the calibration tolerance
- `approximate`: calibration stayed bounded but did not converge inside the tolerance
- `failed`: calibration could not be enforced or prompt token counts could not be confirmed

`failed` context-fill samples remain in raw artifacts for transparency, but they are excluded from strict calibrated aggregates.

## Metrics

The main artifact metrics use explicit names:

- `load_duration_ms`: model load time reported by Ollama, especially relevant for true cold-start runs.
- `prompt_tokens_per_second`: prompt-side throughput from Ollama prompt evaluation counters.
- `generation_tokens_per_second`: generation-side throughput from Ollama eval counters.
- `ttft_ms`: time to first streamed model emission for TTFT scenarios.
- `stream_emission_count`: number of non-empty streamed output emissions observed in TTFT scenarios.
- `stream_duration_ms`: time from the first non-empty streamed output emission to the last non-empty streamed output emission.
- `stream_emission_interval_ms_median`: median gap between consecutive non-empty streamed output emissions.
- `stream_emission_interval_ms_p95`: nearest-rank p95 gap between consecutive non-empty streamed output emissions.
- `stream_output_units_per_second`: non-empty streamed output emissions per second across `stream_duration_ms`.

The TTFT family keeps its first-emission semantics, but includes one small structured-output probe that asks for several short labeled lines so stream-shape metrics are more likely to observe multiple non-empty emissions.

- `elapsed_ms`: whole-run wall-clock time. For `concurrency-smoke`, this is the overall scenario elapsed time, not the sum or percentile of individual request times.
- `concurrency_request_elapsed_ms_p50`: p50 elapsed time across requests inside one `concurrency-smoke` run.
- `concurrency_request_elapsed_ms_p95`: nearest-rank p95 elapsed time across requests inside one `concurrency-smoke` run.
- `concurrency_request_ttft_ms_p50`: p50 TTFT across requests inside one `concurrency-smoke` run, when streamed output is observed.
- `concurrency_request_ttft_ms_p95`: nearest-rank p95 TTFT across requests inside one `concurrency-smoke` run, when streamed output is observed.
- `gpu_telemetry_available`: whether live GPU telemetry was collected for the run.
- `gpu_telemetry_source`: best-effort source used for live GPU telemetry.
- `gpu_backend`: backward-compatible alias of `gpu_telemetry_source` in raw artifacts.
- `peak_gpu_memory_mb`: highest sampled GPU memory use during the run.
- `avg_gpu_util_percent`: average sampled GPU utilization during the run.
- `peak_gpu_util_percent`: highest sampled GPU utilization during the run.
- `sampled_process_count`: count of unique process IDs listed in `sampled_process_ids` for the run.
- `sampled_process_ids`: unique process IDs observed in host resource samples for the run.
- `available_system_memory_mb`: best-effort available host memory snapshot captured immediately before each run starts.
- `system_cpu_load_snapshot`: best-effort host CPU utilization percentage snapshot captured immediately before each run starts.
- `prompt_eval_count`: Ollama-reported prompt evaluation token count, when available.
- `actual_prompt_tokens`: calibrated prompt token count for calibrated context-fill scenarios, when available.
- `target_prompt_tokens`: requested calibrated prompt token target for calibrated context-fill scenarios, when available.

`tokens_per_second` is still stored as a backward-compatible alias of `generation_tokens_per_second`; reports and summaries prefer the generation-specific name.

Stream-shape telemetry is intentionally chunk/emission based. Ollama's streaming API does not provide exact per-token timestamps for these local runs, so the profiler records the client-observed arrival time of each non-empty streamed output chunk after request start. The output unit is recorded as `stream_output_unit: emission` in raw artifacts to avoid implying token-accurate timing. If a stream has one non-empty emission, interval metrics and `stream_output_units_per_second` are `null`; if it has no non-empty emissions, `stream_duration_ms` is also `null`.

`concurrency-smoke` is a small same-machine contention benchmark, not a server-capacity or load-testing benchmark. It runs fixed parallelism scenarios of 2 and 4 overlapping streamed generate requests using local threads, records each request's elapsed time and TTFT when a non-empty streamed output chunk is observed, and reports per-run request-level p50/p95 elapsed and TTFT metrics. The run-level `elapsed_ms` remains whole-run wall time for the concurrent scenario. Each worker opens its own Ollama client instance so the concurrent path avoids sharing one mutable client object across worker threads. It does not estimate max RPS, model queue behavior under sustained load, distributed clients, or saturation capacity.

Live GPU telemetry is optional and best-effort. The current implementation samples `nvidia-smi` when it is available, records unavailable/error notes instead of failing the run when it is not, and does not yet collect live AMD ROCm or Apple Silicon GPU metrics. GPU telemetry uses its own conservative polling interval, currently 0.5 seconds by default, and intermediate process-sampler ticks reuse the most recent GPU sample rather than invoking `nvidia-smi` every time. If `nvidia-smi` is missing, that unavailable result is cached for the rest of the run; command failures may be retried, but not more often than the GPU polling interval. On multi-GPU systems, sampled memory is summed across reported GPUs and utilization is averaged across reported GPUs; raw artifacts include `gpu_telemetry_notes` when this aggregation is used. These values are useful for spotting accelerator memory pressure or possible offload/spill behavior, but they are sampled snapshots rather than a full profiler trace.

Host process telemetry is attributed conservatively. The sampler starts with processes whose names begin with `ollama` and, when process IDs and parent process IDs are available, includes descendants of those Ollama root processes. This catches related runner/worker child processes without broad fuzzy matching. If parent/child metadata is unavailable, the sampler falls back to the root Ollama-name match only. Raw artifacts include `sampled_process_count` and `sampled_process_ids` so you can see what was counted.

Host pressure context is a lightweight pre-run snapshot, not a full monitor. Before each run, the profiler records available system memory and a portable CPU utilization snapshot using `psutil.cpu_percent(interval=None)`. On most platforms that CPU value represents utilization since the previous psutil CPU sample, so it is useful as a coarse "was the machine already busy?" clue rather than a precise load-average trace. Runs may include an advisory `host_pressure_warning` when pre-run CPU is at least 80%, memory use is at least 90%, or available memory is below 1024 MB. These warnings never block or fail a run; they are context for interpreting suspicious results.

Plot export is optional and does not affect profiling runs. `owp export-plots SESSION_DIR` reads `SESSION_DIR/summary.json` and writes dependency-free SVG files under `SESSION_DIR/plots` by default:

- `latency_vs_prompt_size.svg`: TTFT vs prompt size, plus median stream-emission interval when available.
- `throughput_vs_prompt_size.svg`: prompt processing speed and generation speed vs prompt size.
- `README.md`: notes describing source fields and limitations.

The plot helper is intentionally lightweight: it does not require matplotlib, does not read raw run artifacts, and does not attempt publication layout beyond clear labels, units, and legends. Prompt size uses the first available summary field in this order: `prompt_eval_count_median`, `actual_prompt_tokens_median`, `target_prompt_tokens_median`, then `context_size`. Older summaries or benchmark rows without token-count summaries may therefore plot against context size rather than measured prompt tokens. Stream cadence remains emission/chunk based, not token-timed.

Summary statistics are intentionally conservative:

- `median`: the middle observed value for the included sample set
- `p95`: the 95th percentile using a consistent nearest-rank method
- `n`: the number of samples that qualified for that aggregate

Aggregates use completed runs only, and strict benchmark summaries also respect prep and calibration eligibility flags.

## Validated On

This release candidate was live-validated against a real local Ollama setup on `main`, including `owp doctor` and an interactive bounded `owp profile` run with seed artifacts, append-only raw results, telemetry, phase peaks, and TTFT capture on the chat-based path.

## Release Checklist

Run these before publishing:

```powershell
python -m pytest -q
python -m ollama_workload_profiler doctor
owp profile
```

For the final bounded smoke check, use one local model, one context, and a small benchmark selection such as `smoke` and `ttft`.

## TTFT Notes

TTFT handling is deterministic by policy:

- chat-based TTFT is the most reliable observable path on some local models
- generate-based TTFT may legitimately produce no first streamed token on some models, and this is handled as a policy outcome rather than a runtime failure

## Current limits

V1 is intentionally local and conservative:

- one model per session
- one or more contexts per session
- consecutive execution only
- fixed artifact filenames per session
- benchmark families are deterministic and still intentionally local-first rather than distributed; `concurrency-smoke` is limited to same-machine contention at parallelism 2 and 4
- environment metadata such as accelerator details, VRAM, live GPU telemetry, and some Ollama runtime/model fields are best-effort and may be partially unavailable depending on platform and local tooling
