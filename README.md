# Ollama Workload Profiler

`ollama-workload-profiler` is a CLI-first local benchmarking and profiling tool for Ollama on a single machine.

It is designed for local serving and workload comparisons, not for scientific benchmarking and not for model-quality evaluation. This project does not run standardized quality suites such as MMLU or GSM8K. The goal is to make local Ollama runs more repeatable, more honest, and easier to explain in a blog post or hardware write-up.

## Install

The repo ships with pinned dependencies in both `requirements.txt` and `pyproject.toml`.
The bootstrap script creates or reuses `.venv`, installs dependencies, installs the project itself into that venv with an editable install, and prints a plain status line for each step.

```powershell
python scripts/bootstrap.py
```

Because the project is installed in editable mode, local changes under `src/` are reflected immediately without setting a custom `PYTHONPATH`.

## Usage

Bootstrap first, then either activate the environment or call its interpreter directly:

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

Run help first if you are new to the CLI:

```powershell
python -m ollama_workload_profiler --help
python -m ollama_workload_profiler doctor
owp --help
owp doctor
owp profile --model llama3.2 --contexts 4096,8192 --benchmark-types smoke,use-case-profiles
```

`owp doctor` reports platform, Python, venv status, dependency status, Ollama binary detection, Ollama reachability, local model availability, and remediation hints when something is missing.

`owp profile` supports an interactive flow by default: it can prompt for model, contexts, and benchmark families, shows a benchmark budget summary, and asks for confirmation before execution. You can still pass `--model`, `--contexts`, and `--benchmark-types` directly when scripting.

The current CLI accepts `smoke`, `cold-warm`, `prompt-scaling`, `context-scaling`, `output-scaling`, `use-case-profiles`, `ttft`, and `stress`.

Results are written to timestamped directories under `results/` by default.

## Quick recipes

Run a quick smoke plus TTFT session:

```powershell
owp profile --model llama3.2 --contexts 4096 --benchmark-types smoke,ttft --yes
```

Compare two completed sessions:

```powershell
owp compare results/session-a results/session-b
```

Compare with strict comparability checks and full metric tables:

```powershell
owp compare results/session-a results/session-b --strict --all-metrics
```

Compare two completed sessions without rerunning benchmarks:

```powershell
owp compare results/session-2026-04-20_T10-00-00Z results/session-2026-04-21_T10-00-00Z
owp compare results/session-a results/session-b --format json --output compare.json
owp compare results/session-a results/session-b --strict
```

The comparison direction is `baseline -> candidate`; deltas are candidate minus baseline. `--strict` exits nonzero for strict-blocking comparability issues. `owp compare` is read-only. It reports deltas from existing `summary.json`, `plan.json`, and `environment.json` artifacts, warns when sessions are not directly comparable under the same benchmark contract, and does not make statistical significance claims. Warnings are comparability/context signals, not proof of a meaningful performance difference.

## Benchmark semantics

The profiler now treats benchmark policy as first-class session data. Requested execution settings such as `seed`, `temperature`, `top_p`, `repetitions`, and warmup behavior are written into the session artifacts so runs are easier to reproduce and explain later.

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

The key metrics are reported with explicit names:

- `load_duration_ms`: model load time reported by Ollama. This is especially important for true cold-start runs.
- `prompt_tokens_per_second`: prompt-side throughput from Ollama prompt evaluation counters.
- `generation_tokens_per_second`: generation-side throughput from Ollama eval counters.
- `ttft_ms`: time to first streamed model emission for TTFT scenarios.

`tokens_per_second` is still stored as a backward-compatible alias of `generation_tokens_per_second`, but reports and summaries prefer the generation-specific name.

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
- benchmark families are deterministic and still intentionally local-first rather than distributed or concurrent
- environment metadata such as accelerator details, VRAM, and some Ollama runtime/model fields are best-effort and may be partially unavailable depending on platform and local tooling
