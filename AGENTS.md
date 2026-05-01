# Repository Guidelines

## Project overview
- `ollama-workload-profiler` is a CLI-first local benchmarking and profiling tool for Ollama on one machine.
- Main stack: Python 3.11+, Typer CLI, pytest, ruff, psutil, httpx, and pydantic.
- Main entry points:
  - `src/ollama_workload_profiler/cli.py`
  - `src/ollama_workload_profiler/__main__.py`
  - `scripts/bootstrap.py`

## Repo layout
- `src/ollama_workload_profiler/` main package.
- `src/ollama_workload_profiler/benchmarks/` benchmark families.
- `src/ollama_workload_profiler/reporting/` compare output, markdown, plots, and artifact writers.
- `src/ollama_workload_profiler/metrics/` telemetry and sampling helpers.
- `src/ollama_workload_profiler/models/` plan, result, summary, and verdict schemas.
- `tests/` pytest suite.
- `scripts/bootstrap.py` creates/reuses the local venv and editable install.
- `scripts/run_pre_push_checks.py` is the local pre-push runner; `.githooks/pre-push` points to it.
- `docs/superpowers/specs/` and `docs/superpowers/plans/` hold historical design and implementation notes.
- Generated or local-only paths are read-only for context unless a task explicitly asks to update generated output:
  - `results/`
  - `dist/`
  - `.venv/`
  - `.worktrees/`
  - `.pytest_cache/`
  - `.ruff_cache/`

## Commands
- No repo-local CI workflow files were present in this checkout; use the local commands below as the source of truth.
- Run all commands from the repo root unless noted.

Ask first:
```powershell
python scripts/bootstrap.py
```
- Creates or reuses `.venv/`, installs pinned dependencies, and may use the network.
- Use for a fresh setup or when the local environment needs repair.

Safe validation:
```powershell
python scripts/run_pre_push_checks.py
python -m ruff check .
python -m compileall -q src tests
python -m pytest -q
python -m ollama_workload_profiler doctor
```
- `scripts/run_pre_push_checks.py` runs Ruff, compileall, then pytest in that order.
- `python -m pytest -q` is the full test suite.
- `python -m ollama_workload_profiler doctor` is read-only, but depends on local Ollama readiness.
- `python -m compileall -q src tests` is bytecode validation only; it does not type-check.
- No dedicated `format`, `type-check`, or `build` command was found. Do not invent one.

Write-producing, only when requested:
```powershell
owp export-plots <session_dir>
```
- Regenerates SVG plot artifacts under `<session_dir>/plots/` for a completed session.
- Run only when generated plot output is part of the requested task.

## Validation before completion
- Run the smallest relevant validation set for the change.
- Report exact commands run and pass/fail results.
- Report checks skipped and why.
- Include a final `git status --short` summary when git is available.

## Safety and approval boundaries
Allowed without asking:
- Read, search, and edit files inside the repo.
- Run safe local validation commands listed in this file.
- Inspect generated artifacts or logs for context.

Ask first:
- Ask before running `python scripts/bootstrap.py`, dependency installs, or dependency upgrades.
- Ask before running networked shell commands.
- Ask before deleting files unless the task directly requests deletion.
- Ask before regenerating generated output unless the task directly requests regeneration.
- Ask before changing lockfiles, package metadata, or environment files unless requested.
- Ask before writing outside the repo or changing CI, release, deployment, infra, or migration-related files.

Never unless explicitly approved:
- Do not run git state-changing commands:
  - `commit`
  - `push`
  - `force-push`
  - `rebase`
  - `reset`
  - destructive checkout commands
  - commands using `--no-verify`
- Do not modify secrets, credentials, private keys, real environment files, or credential files. Safe templates or examples may be edited only if they do not contain real secrets.
- Do not hide failing tests, lint, or CI by weakening checks.
- Do not run destructive cleanup or any apply, publish, release, or deploy command.

## Testing conventions
- Tests live in `tests/` and use pytest.
- CLI tests use `typer.testing.CliRunner`; keep command help and output assertions stable.
- Use `tmp_path` and `monkeypatch` for isolation instead of live Ollama calls in unit tests.
- Match tests to the changed area:
  - CLI behavior: `tests/test_cli_validation.py`
  - environment/bootstrap logic: `tests/test_env_check.py`, `tests/test_bootstrap.py`
  - pre-push behavior: `tests/test_pre_push.py`
  - compare/reporting semantics: `tests/test_compare.py`, `tests/test_reporting.py`
  - session execution and artifacts: `tests/test_session.py`
  - README command snippets: `tests/test_readme.py`
- `pyproject.toml` sets `pythonpath = ["src"]`, so tests import the package directly.

## Architecture or code conventions
- Keep `requirements.txt` and `pyproject.toml` dependency pins aligned; tests assert they match exactly.
- Preserve `benchmark_methodology_version` semantics; bump it only when the benchmark contract changes, not for doc or CLI text changes.
- Keep `plan.json`, `environment.json`, and `summary.json` consistent when touching session or reporting code.
- `compare` is read-only and compares `baseline -> candidate`; do not change that direction without updating tests and docs.
- Keep session artifact filenames stable:
  - `plan.json`
  - `expanded_plan.json`
  - `environment.json`
  - `raw.jsonl`
  - `raw.csv`
  - `summary.json`
  - `report.md`
- `scripts/bootstrap.py` expects editable installs from `src/`; keep its status messages and detection behavior in sync with the tests.
- Generated output should stay under `results/` or a session-local output directory; do not write generated files into source trees.

## Docs
- Read `README.md` before changing install, CLI, or usage behavior.
- Read relevant files under `docs/superpowers/specs/` when changing benchmark semantics, comparison rules, or artifact formats.
- Update `README.md` and affected tests when user-facing commands, flags, or artifact schemas change.
- There is no separate repo architecture manual in this checkout; keep docs changes focused on user-visible behavior.

## Nested AGENTS.md
- A nested `AGENTS.md` is probably unnecessary here because the repo uses one Python stack and one validation pipeline.
- Revisit nested guidance only if a subtree gains different tooling, ownership, or approval rules.
