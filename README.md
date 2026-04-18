# Ollama Workload Profiler

`ollama-workload-profiler` is a CLI-first local benchmarking and profiling tool for Ollama.

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
