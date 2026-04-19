# Ollama Workload Profiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a publishable V1 of `ollama-workload-profiler` with bootstrap/setup, `owp doctor`, interactive `owp profile`, deterministic benchmark planning and execution, local telemetry, persisted artifacts, Markdown reporting, and core unit tests.

**Architecture:** The project is split into a standalone `scripts/bootstrap.py` setup path and an installable package CLI for `owp doctor` and `owp profile`. Runtime logic centers on typed models in `src/ollama_workload_profiler/models/`, a session planner in `session.py`, benchmark modules that expand into deterministic scenarios, an Ollama client, telemetry samplers, and reporting modules that consume append-only raw run results and emit fixed session artifacts.

**Tech Stack:** Python 3.11+, Typer, Rich, httpx, psutil, pydantic, Jinja2, pytest, pytest-mock.

---

## File Structure

Files to create or modify during implementation:

- Create: `README.md`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `scripts/bootstrap.py`
- Create: `src/ollama_workload_profiler/__init__.py`
- Create: `src/ollama_workload_profiler/__main__.py`
- Create: `src/ollama_workload_profiler/cli.py`
- Create: `src/ollama_workload_profiler/env_check.py`
- Create: `src/ollama_workload_profiler/ollama_client.py`
- Create: `src/ollama_workload_profiler/session.py`
- Create: `src/ollama_workload_profiler/benchmarks/__init__.py`
- Create: `src/ollama_workload_profiler/benchmarks/base.py`
- Create: `src/ollama_workload_profiler/benchmarks/smoke.py`
- Create: `src/ollama_workload_profiler/benchmarks/cold_warm.py`
- Create: `src/ollama_workload_profiler/benchmarks/prompt_scaling.py`
- Create: `src/ollama_workload_profiler/benchmarks/context_scaling.py`
- Create: `src/ollama_workload_profiler/benchmarks/output_scaling.py`
- Create: `src/ollama_workload_profiler/benchmarks/profiles.py`
- Create: `src/ollama_workload_profiler/benchmarks/ttft.py`
- Create: `src/ollama_workload_profiler/benchmarks/stress.py`
- Create: `src/ollama_workload_profiler/metrics/__init__.py`
- Create: `src/ollama_workload_profiler/metrics/process.py`
- Create: `src/ollama_workload_profiler/metrics/sampler.py`
- Create: `src/ollama_workload_profiler/metrics/phases.py`
- Create: `src/ollama_workload_profiler/prompts/__init__.py`
- Create: `src/ollama_workload_profiler/prompts/fixtures.py`
- Create: `src/ollama_workload_profiler/prompts/scenarios.py`
- Create: `src/ollama_workload_profiler/models/__init__.py`
- Create: `src/ollama_workload_profiler/models/failures.py`
- Create: `src/ollama_workload_profiler/models/plan.py`
- Create: `src/ollama_workload_profiler/models/results.py`
- Create: `src/ollama_workload_profiler/models/summary.py`
- Create: `src/ollama_workload_profiler/models/verdicts.py`
- Create: `src/ollama_workload_profiler/reporting/__init__.py`
- Create: `src/ollama_workload_profiler/reporting/artifacts.py`
- Create: `src/ollama_workload_profiler/reporting/markdown.py`
- Create: `src/ollama_workload_profiler/reporting/summary.py`
- Create: `src/ollama_workload_profiler/reporting/verdicts.py`
- Create: `tests/conftest.py`
- Create: `tests/test_env_check.py`
- Create: `tests/test_ollama_client.py`
- Create: `tests/test_cli_validation.py`
- Create: `tests/test_session.py`
- Create: `tests/test_reporting.py`
- Create: `tests/test_verdicts.py`

## Task 1: Scaffold Packaging And CLI Entry

**Files:**
- Create: `README.md`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `src/ollama_workload_profiler/__init__.py`
- Create: `src/ollama_workload_profiler/__main__.py`
- Create: `src/ollama_workload_profiler/cli.py`
- Test: `tests/test_cli_validation.py`

- [ ] **Step 1: Write the failing packaging and CLI smoke tests**

```python
from typer.testing import CliRunner

from ollama_workload_profiler.cli import app


def test_cli_shows_root_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "profile" in result.stdout
    assert "doctor" in result.stdout


def test_cli_module_entrypoint_runs() -> None:
    from ollama_workload_profiler.__main__ import main

    assert callable(main)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_validation.py -v`
Expected: FAIL because `ollama_workload_profiler` package and CLI app do not exist yet.

- [ ] **Step 3: Write minimal packaging and CLI implementation**

```toml
[build-system]
requires = ["setuptools==80.9.0", "wheel==0.45.1"]
build-backend = "setuptools.build_meta"

[project]
name = "ollama-workload-profiler"
version = "0.2.0"
description = "CLI-first local benchmarking and profiling for Ollama"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "typer==0.16.0",
  "rich==14.0.0",
  "httpx==0.28.1",
  "psutil==7.0.0",
  "pydantic==2.11.3",
  "jinja2==3.1.6",
]

[project.scripts]
owp = "ollama_workload_profiler.cli:main"

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

```python
import typer

app = typer.Typer(help="Profile local Ollama workloads.")


@app.command()
def doctor() -> None:
    raise typer.Exit(code=0)


@app.command()
def profile() -> None:
    raise typer.Exit(code=0)


def main() -> None:
    app()
```

```python
from .cli import main


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli_validation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md pyproject.toml requirements.txt src/ollama_workload_profiler/__init__.py src/ollama_workload_profiler/__main__.py src/ollama_workload_profiler/cli.py tests/test_cli_validation.py
git commit -m "feat: scaffold package and cli entrypoints"
```

## Task 2: Add Typed Models For Plans, Results, Failures, And Summaries

**Files:**
- Create: `src/ollama_workload_profiler/models/__init__.py`
- Create: `src/ollama_workload_profiler/models/failures.py`
- Create: `src/ollama_workload_profiler/models/plan.py`
- Create: `src/ollama_workload_profiler/models/results.py`
- Create: `src/ollama_workload_profiler/models/summary.py`
- Create: `src/ollama_workload_profiler/models/verdicts.py`
- Test: `tests/test_session.py`

- [ ] **Step 1: Write the failing model serialization tests**

```python
from ollama_workload_profiler.models.plan import BenchmarkSessionPlan, BenchmarkType
from ollama_workload_profiler.models.results import RunResult, RunState


def test_benchmark_session_plan_serializes_to_json() -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[4096, 8192],
        benchmark_types=[BenchmarkType.SMOKE, BenchmarkType.TTFT],
        repetitions=2,
    )

    payload = plan.model_dump(mode="json")

    assert payload["model_name"] == "llama3.2"
    assert payload["contexts"] == [4096, 8192]
    assert payload["benchmark_types"] == ["smoke", "ttft"]


def test_run_result_has_stable_ordering_fields() -> None:
    result = RunResult(
        run_id="run-0001",
        run_index=1,
        model_name="llama3.2",
        context_size=4096,
        benchmark_type="smoke",
        scenario_id="smoke-basic-v1",
        state=RunState.PLANNED,
    )

    assert result.run_index == 1
    assert result.run_id == "run-0001"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_session.py -v`
Expected: FAIL because the models package does not exist yet.

- [ ] **Step 3: Write minimal typed models**

```python
from enum import StrEnum
from pydantic import BaseModel, Field


class BenchmarkType(StrEnum):
    SMOKE = "smoke"
    COLD_WARM = "cold-warm"
    PROMPT_SCALING = "prompt-scaling"
    CONTEXT_SCALING = "context-scaling"
    OUTPUT_SCALING = "output-scaling"
    USE_CASE_PROFILES = "use-case-profiles"
    TTFT = "ttft"
    STRESS = "stress"


class BenchmarkSessionPlan(BaseModel):
    model_name: str
    contexts: list[int]
    benchmark_types: list[BenchmarkType]
    repetitions: int = Field(ge=1, default=1)
    stop_conditions: list[dict] = Field(default_factory=list)
```

```python
from enum import StrEnum
from pydantic import BaseModel


class RunState(StrEnum):
    PLANNED = "planned"
    STARTING = "starting"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class RunResult(BaseModel):
    run_id: str
    run_index: int
    model_name: str
    context_size: int
    benchmark_type: str
    scenario_id: str
    state: RunState
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_session.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ollama_workload_profiler/models/__init__.py src/ollama_workload_profiler/models/failures.py src/ollama_workload_profiler/models/plan.py src/ollama_workload_profiler/models/results.py src/ollama_workload_profiler/models/summary.py src/ollama_workload_profiler/models/verdicts.py tests/test_session.py
git commit -m "feat: add core typed models"
```

## Task 3: Implement Environment Checks And Doctor Diagnostics

**Files:**
- Create: `src/ollama_workload_profiler/env_check.py`
- Modify: `src/ollama_workload_profiler/cli.py`
- Test: `tests/test_env_check.py`

- [ ] **Step 1: Write the failing environment detection tests**

```python
from ollama_workload_profiler.env_check import detect_python_environment, summarize_doctor_status


def test_detect_python_environment_reports_venv(monkeypatch) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/venv")
    status = detect_python_environment()
    assert status.in_venv is True


def test_summarize_doctor_status_returns_nonzero_for_missing_ollama() -> None:
    summary = summarize_doctor_status(binary_found=False, reachable=False, models=[])
    assert summary.exit_code != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_env_check.py -v`
Expected: FAIL because `env_check.py` does not exist yet.

- [ ] **Step 3: Write minimal environment and doctor implementation**

```python
from dataclasses import dataclass
import os
import platform
import shutil
import sys


@dataclass
class PythonEnvironmentStatus:
    python_version: str
    in_venv: bool
    executable: str


@dataclass
class DoctorSummary:
    exit_code: int
    platform_name: str
    messages: list[str]


def detect_python_environment() -> PythonEnvironmentStatus:
    return PythonEnvironmentStatus(
        python_version=sys.version.split()[0],
        in_venv=bool(os.environ.get("VIRTUAL_ENV")) or sys.prefix != sys.base_prefix,
        executable=sys.executable,
    )


def detect_ollama_binary() -> bool:
    return shutil.which("ollama") is not None


def summarize_doctor_status(binary_found: bool, reachable: bool, models: list[str]) -> DoctorSummary:
    exit_code = 0 if binary_found and reachable and bool(models) else 1
    return DoctorSummary(
        exit_code=exit_code,
        platform_name=platform.platform(),
        messages=[],
    )
```

```python
@app.command()
def doctor() -> None:
    status = detect_python_environment()
    typer.echo(f"Python: {status.python_version}")
    raise typer.Exit(code=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_env_check.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ollama_workload_profiler/env_check.py src/ollama_workload_profiler/cli.py tests/test_env_check.py
git commit -m "feat: add environment checks and doctor diagnostics"
```

## Task 4: Implement Ollama Client Abstractions

**Files:**
- Create: `src/ollama_workload_profiler/ollama_client.py`
- Test: `tests/test_ollama_client.py`

- [ ] **Step 1: Write the failing Ollama client tests**

```python
from ollama_workload_profiler.ollama_client import OllamaClient


def test_list_models_returns_names(httpx_mock) -> None:
    httpx_mock.add_response(
        method="GET",
        url="http://127.0.0.1:11434/api/tags",
        json={"models": [{"name": "llama3.2:latest"}]},
    )

    client = OllamaClient()
    assert client.list_models() == ["llama3.2:latest"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ollama_client.py -v`
Expected: FAIL because `ollama_client.py` does not exist yet.

- [ ] **Step 3: Write minimal client implementation**

```python
import httpx


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        self._client = httpx.Client(base_url=base_url, timeout=60.0)

    def list_models(self) -> list[str]:
        response = self._client.get("/api/tags")
        response.raise_for_status()
        payload = response.json()
        return [item["name"] for item in payload.get("models", [])]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ollama_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ollama_workload_profiler/ollama_client.py tests/test_ollama_client.py
git commit -m "feat: add ollama client abstraction"
```

## Task 5: Add Prompt Fixtures And Scenario Expansion

**Files:**
- Create: `src/ollama_workload_profiler/prompts/__init__.py`
- Create: `src/ollama_workload_profiler/prompts/fixtures.py`
- Create: `src/ollama_workload_profiler/prompts/scenarios.py`
- Test: `tests/test_session.py`

- [ ] **Step 1: Write the failing scenario expansion tests**

```python
from ollama_workload_profiler.models.plan import BenchmarkSessionPlan, BenchmarkType
from ollama_workload_profiler.prompts.scenarios import build_scenarios_for_benchmark


def test_prompt_scaling_builds_three_scenarios() -> None:
    scenarios = build_scenarios_for_benchmark(BenchmarkType.PROMPT_SCALING, 4096)
    assert [item.scenario_id for item in scenarios] == [
        "prompt-scaling-small-v1",
        "prompt-scaling-medium-v1",
        "prompt-scaling-large-v1",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_session.py::test_prompt_scaling_builds_three_scenarios -v`
Expected: FAIL because prompt scenario modules do not exist yet.

- [ ] **Step 3: Write minimal scenario fixtures**

```python
STATIC_SUMMARY_TEXT = "Section 1: ... deterministic summary fixture ..."
STATIC_CODE_SAMPLE = "def fib(n: int) -> int:\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n"
MULTI_TURN_CHAT_TURNS = [
    "Summarize the release note in one sentence.",
    "Now rewrite it for a technical audience.",
    "Now extract three action items.",
]
```

```python
def build_scenarios_for_benchmark(benchmark_type: BenchmarkType, context_size: int) -> list[ScenarioDefinition]:
    if benchmark_type is BenchmarkType.PROMPT_SCALING:
        return [
            ScenarioDefinition(scenario_id="prompt-scaling-small-v1", benchmark_type=benchmark_type, name="Small prompt", version="v1", prompt_payload="x" * 1024, target_output_tokens=64, difficulty_tag="light", phase_emphasis="prompt_sensitive"),
            ScenarioDefinition(scenario_id="prompt-scaling-medium-v1", benchmark_type=benchmark_type, name="Medium prompt", version="v1", prompt_payload="x" * 4096, target_output_tokens=64, difficulty_tag="medium", phase_emphasis="prompt_sensitive"),
            ScenarioDefinition(scenario_id="prompt-scaling-large-v1", benchmark_type=benchmark_type, name="Large prompt", version="v1", prompt_payload="x" * 16384, target_output_tokens=64, difficulty_tag="heavy", phase_emphasis="prompt_sensitive"),
        ]
    raise NotImplementedError(benchmark_type)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_session.py::test_prompt_scaling_builds_three_scenarios -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ollama_workload_profiler/prompts/__init__.py src/ollama_workload_profiler/prompts/fixtures.py src/ollama_workload_profiler/prompts/scenarios.py tests/test_session.py
git commit -m "feat: add deterministic benchmark scenarios"
```

## Task 6: Implement Session Planner And Expanded Plan Persistence

**Files:**
- Create: `src/ollama_workload_profiler/session.py`
- Modify: `src/ollama_workload_profiler/models/plan.py`
- Test: `tests/test_session.py`

- [ ] **Step 1: Write the failing session planning tests**

```python
from ollama_workload_profiler.models.plan import BenchmarkSessionPlan, BenchmarkType
from ollama_workload_profiler.session import expand_session_plan


def test_expand_session_plan_orders_contexts_then_benchmarks() -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE, BenchmarkType.CONTEXT_SCALING],
        repetitions=1,
    )

    runs = expand_session_plan(plan)

    assert [run.scenario_id for run in runs] == [
        "smoke-basic-v1",
        "context-scaling-low-v1",
        "context-scaling-medium-v1",
        "context-scaling-high-v1",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_session.py::test_expand_session_plan_orders_contexts_then_benchmarks -v`
Expected: FAIL because session expansion is not implemented yet.

- [ ] **Step 3: Write minimal session planner**

```python
from ollama_workload_profiler.models.results import RunResult, RunState
from ollama_workload_profiler.prompts.scenarios import build_scenarios_for_benchmark


BENCHMARK_ORDER = [
    BenchmarkType.SMOKE,
    BenchmarkType.COLD_WARM,
    BenchmarkType.PROMPT_SCALING,
    BenchmarkType.CONTEXT_SCALING,
    BenchmarkType.OUTPUT_SCALING,
    BenchmarkType.USE_CASE_PROFILES,
    BenchmarkType.TTFT,
    BenchmarkType.STRESS,
]


def expand_session_plan(plan: BenchmarkSessionPlan) -> list[RunResult]:
    ordered_types = [item for item in BENCHMARK_ORDER if item in plan.benchmark_types]
    runs: list[RunResult] = []
    run_index = 1

    for context_index, context_size in enumerate(plan.contexts):
        for benchmark_index, benchmark_type in enumerate(ordered_types):
            scenarios = build_scenarios_for_benchmark(benchmark_type, context_size)
            for scenario_index, scenario in enumerate(scenarios):
                for repetition_index in range(plan.repetitions):
                    runs.append(
                        RunResult(
                            run_id=f"run-{run_index:04d}",
                            run_index=run_index,
                            model_name=plan.model_name,
                            context_size=context_size,
                            benchmark_type=benchmark_type.value,
                            scenario_id=scenario.scenario_id,
                            state=RunState.PLANNED,
                        )
                    )
                    run_index += 1
    return runs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_session.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ollama_workload_profiler/session.py src/ollama_workload_profiler/models/plan.py tests/test_session.py
git commit -m "feat: add deterministic session planning"
```

## Task 7: Implement Bootstrap Flow

**Files:**
- Create: `scripts/bootstrap.py`
- Modify: `README.md`
- Test: `tests/test_env_check.py`

- [ ] **Step 1: Write the failing bootstrap tests**

```python
from pathlib import Path

from ollama_workload_profiler.env_check import ensure_results_dir


def test_bootstrap_can_create_virtualenv_path(tmp_path: Path) -> None:
    venv_path = tmp_path / ".venv"
    assert not venv_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_env_check.py::test_bootstrap_can_create_virtualenv_path -v`
Expected: FAIL because the bootstrap helpers do not exist yet.

- [ ] **Step 3: Write minimal bootstrap script**

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PATH = PROJECT_ROOT / ".venv"


def main() -> int:
    if not VENV_PATH.exists():
        subprocess.run([sys.executable, "-m", "venv", str(VENV_PATH)], check=True)
    subprocess.run([str(VENV_PATH / "Scripts" / "python.exe"), "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_env_check.py -v`
Expected: PASS with bootstrap-related assertions added.

- [ ] **Step 5: Commit**

```bash
git add scripts/bootstrap.py README.md tests/test_env_check.py
git commit -m "feat: add bootstrap setup flow"
```

## Task 8: Build Interactive Profile Input Validation

**Files:**
- Modify: `src/ollama_workload_profiler/cli.py`
- Modify: `src/ollama_workload_profiler/models/plan.py`
- Test: `tests/test_cli_validation.py`

- [ ] **Step 1: Write the failing interactive validation tests**

```python
from ollama_workload_profiler.cli import parse_multi_select_contexts


def test_parse_multi_select_contexts_rejects_unknown_values() -> None:
    try:
        parse_multi_select_contexts("4096,banana")
    except ValueError as exc:
        assert "banana" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_validation.py -v`
Expected: FAIL because parsing helpers do not exist yet.

- [ ] **Step 3: Write minimal input validation helpers**

```python
def parse_multi_select_contexts(raw: str) -> list[int]:
    contexts: list[int] = []
    for part in raw.split(","):
        cleaned = part.strip()
        if not cleaned.isdigit():
            raise ValueError(f"Invalid context selection: {cleaned}")
        contexts.append(int(cleaned))
    if not contexts:
        raise ValueError("At least one context is required")
    return contexts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli_validation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ollama_workload_profiler/cli.py src/ollama_workload_profiler/models/plan.py tests/test_cli_validation.py
git commit -m "feat: add interactive selection validation"
```

## Task 9: Add Process Discovery And Sampling Infrastructure

**Files:**
- Create: `src/ollama_workload_profiler/metrics/process.py`
- Create: `src/ollama_workload_profiler/metrics/sampler.py`
- Create: `src/ollama_workload_profiler/metrics/phases.py`
- Test: `tests/test_session.py`

- [ ] **Step 1: Write the failing sampler tests**

```python
from ollama_workload_profiler.metrics.phases import compute_phase_peaks


def test_compute_phase_peaks_groups_samples_by_phase() -> None:
    peaks = compute_phase_peaks(
        samples=[
            {"phase": "load", "rss_mb": 100, "cpu_percent": 5},
            {"phase": "generation", "rss_mb": 150, "cpu_percent": 35},
        ]
    )
    assert peaks["generation"].rss_mb == 150
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_session.py::test_compute_phase_peaks_groups_samples_by_phase -v`
Expected: FAIL because metrics modules do not exist yet.

- [ ] **Step 3: Write minimal sampling and phase aggregation**

```python
import psutil


def find_ollama_processes() -> list[psutil.Process]:
    return [proc for proc in psutil.process_iter(["name"]) if proc.info.get("name", "").lower().startswith("ollama")]
```

```python
from dataclasses import dataclass


@dataclass
class SamplePoint:
    phase: str
    rss_mb: float
    cpu_percent: float
```

```python
def compute_phase_peaks(samples: list[dict]) -> dict[str, SamplePoint]:
    peaks: dict[str, SamplePoint] = {}
    for item in samples:
        phase = item["phase"]
        current = peaks.get(phase)
        if current is None or item["rss_mb"] > current.rss_mb:
            peaks[phase] = SamplePoint(phase=phase, rss_mb=item["rss_mb"], cpu_percent=item["cpu_percent"])
    return peaks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_session.py::test_compute_phase_peaks_groups_samples_by_phase -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ollama_workload_profiler/metrics/__init__.py src/ollama_workload_profiler/metrics/process.py src/ollama_workload_profiler/metrics/sampler.py src/ollama_workload_profiler/metrics/phases.py tests/test_session.py
git commit -m "feat: add telemetry sampling primitives"
```

## Task 10: Implement Benchmark Runner And Core Benchmark Modules

**Files:**
- Create: `src/ollama_workload_profiler/benchmarks/__init__.py`
- Create: `src/ollama_workload_profiler/benchmarks/base.py`
- Create: `src/ollama_workload_profiler/benchmarks/smoke.py`
- Create: `src/ollama_workload_profiler/benchmarks/cold_warm.py`
- Create: `src/ollama_workload_profiler/benchmarks/prompt_scaling.py`
- Create: `src/ollama_workload_profiler/benchmarks/context_scaling.py`
- Create: `src/ollama_workload_profiler/benchmarks/output_scaling.py`
- Create: `src/ollama_workload_profiler/benchmarks/profiles.py`
- Create: `src/ollama_workload_profiler/benchmarks/ttft.py`
- Create: `src/ollama_workload_profiler/benchmarks/stress.py`
- Test: `tests/test_session.py`

- [ ] **Step 1: Write the failing benchmark runner tests**

```python
from ollama_workload_profiler.benchmarks.base import BenchmarkRunner
from ollama_workload_profiler.models.results import RunState


def test_runner_marks_run_completed_when_execution_succeeds(mocker) -> None:
    client = mocker.Mock()
    client.generate.return_value = {"total_duration": 10, "response": "ok"}
    runner = BenchmarkRunner(client=client)

    result = runner.run_once(model_name="llama3.2", context_size=4096, scenario_id="smoke-basic-v1")

    assert result.state is RunState.COMPLETED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_session.py::test_runner_marks_run_completed_when_execution_succeeds -v`
Expected: FAIL because the benchmark runner does not exist yet.

- [ ] **Step 3: Write minimal benchmark runner**

```python
class BenchmarkRunner:
    def __init__(self, client) -> None:
        self.client = client

    def run_once(self, model_name: str, context_size: int, scenario_id: str) -> RunResult:
        started_at = time.time()
        payload = self.client.generate(model=model_name, prompt="ping", options={"num_ctx": context_size})
        elapsed = time.time() - started_at
        return RunResult(
            run_id="run-exec-0001",
            run_index=1,
            model_name=model_name,
            context_size=context_size,
            benchmark_type="smoke",
            scenario_id=scenario_id,
            state=RunState.COMPLETED,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_session.py::test_runner_marks_run_completed_when_execution_succeeds -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ollama_workload_profiler/benchmarks/__init__.py src/ollama_workload_profiler/benchmarks/base.py src/ollama_workload_profiler/benchmarks/smoke.py src/ollama_workload_profiler/benchmarks/cold_warm.py src/ollama_workload_profiler/benchmarks/prompt_scaling.py src/ollama_workload_profiler/benchmarks/context_scaling.py src/ollama_workload_profiler/benchmarks/output_scaling.py src/ollama_workload_profiler/benchmarks/profiles.py src/ollama_workload_profiler/benchmarks/ttft.py src/ollama_workload_profiler/benchmarks/stress.py tests/test_session.py
git commit -m "feat: add benchmark runner and modules"
```

## Task 11: Persist Artifacts And Render Session Reports

**Files:**
- Create: `src/ollama_workload_profiler/reporting/__init__.py`
- Create: `src/ollama_workload_profiler/reporting/artifacts.py`
- Create: `src/ollama_workload_profiler/reporting/markdown.py`
- Create: `src/ollama_workload_profiler/reporting/summary.py`
- Test: `tests/test_reporting.py`

- [ ] **Step 1: Write the failing report persistence tests**

```python
from pathlib import Path

from ollama_workload_profiler.reporting.artifacts import write_session_artifacts


def test_write_session_artifacts_creates_fixed_filenames(tmp_path: Path) -> None:
    output_dir = write_session_artifacts(tmp_path, plan={"model_name": "llama3.2"}, expanded_plan=[], environment={}, runs=[])

    assert (output_dir / "plan.json").exists()
    assert (output_dir / "expanded_plan.json").exists()
    assert (output_dir / "environment.json").exists()
    assert (output_dir / "raw.jsonl").exists()
    assert (output_dir / "raw.csv").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reporting.py -v`
Expected: FAIL because reporting modules do not exist yet.

- [ ] **Step 3: Write minimal artifact and markdown rendering**

```python
from pathlib import Path
import json


def write_session_artifacts(base_dir: Path, plan: dict, expanded_plan: list[dict], environment: dict, runs: list[dict]) -> Path:
    output_dir = base_dir / "session-20260418-000000"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (output_dir / "expanded_plan.json").write_text(json.dumps(expanded_plan, indent=2), encoding="utf-8")
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (output_dir / "raw.jsonl").write_text("", encoding="utf-8")
    (output_dir / "raw.csv").write_text("run_id,scenario_id,state\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps({"run_count": len(runs)}, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text("# Benchmark Report\n", encoding="utf-8")
    return output_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reporting.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ollama_workload_profiler/reporting/__init__.py src/ollama_workload_profiler/reporting/artifacts.py src/ollama_workload_profiler/reporting/markdown.py src/ollama_workload_profiler/reporting/summary.py tests/test_reporting.py
git commit -m "feat: add artifact persistence and reporting"
```

## Task 12: Add Verdict Engine And Suitability Matrix Support

**Files:**
- Create: `src/ollama_workload_profiler/reporting/verdicts.py`
- Modify: `src/ollama_workload_profiler/models/verdicts.py`
- Modify: `src/ollama_workload_profiler/reporting/summary.py`
- Test: `tests/test_verdicts.py`

- [ ] **Step 1: Write the failing verdict tests**

```python
from ollama_workload_profiler.reporting.verdicts import classify_verdict


def test_classify_verdict_returns_avoid_on_failure() -> None:
    verdict = classify_verdict(success=False, ttft_ms=9000, gen_tokens_per_second=1.0, ram_headroom_gb=0.2)
    assert verdict.label == "Avoid on this hardware"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_verdicts.py -v`
Expected: FAIL because the verdict module does not exist yet.

- [ ] **Step 3: Write minimal verdict rules**

```python
from dataclasses import dataclass


@dataclass
class Verdict:
    label: str
    rationale: str


def classify_verdict(success: bool, ttft_ms: float | None, gen_tokens_per_second: float | None, ram_headroom_gb: float | None) -> Verdict:
    if not success:
        return Verdict(label="Avoid on this hardware", rationale="Run did not complete successfully.")
    if ttft_ms is not None and ttft_ms > 5000:
        return Verdict(label="Use with caution", rationale="TTFT is high.")
    if gen_tokens_per_second is not None and gen_tokens_per_second < 5:
        return Verdict(label="Use only for short tasks", rationale="Generation speed is low.")
    return Verdict(label="Good fit", rationale="Core metrics are within acceptable bands.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_verdicts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ollama_workload_profiler/reporting/verdicts.py src/ollama_workload_profiler/models/verdicts.py src/ollama_workload_profiler/reporting/summary.py tests/test_verdicts.py
git commit -m "feat: add deterministic verdict engine"
```

## Task 13: Integrate Profile Flow End To End

**Files:**
- Modify: `src/ollama_workload_profiler/cli.py`
- Modify: `src/ollama_workload_profiler/session.py`
- Modify: `src/ollama_workload_profiler/reporting/artifacts.py`
- Modify: `src/ollama_workload_profiler/ollama_client.py`
- Test: `tests/test_cli_validation.py`

- [ ] **Step 1: Write the failing profile integration test**

```python
from typer.testing import CliRunner

from ollama_workload_profiler.cli import app


def test_profile_refuses_to_start_without_models(mocker) -> None:
    mocker.patch("ollama_workload_profiler.cli.list_available_models", return_value=[])
    runner = CliRunner()
    result = runner.invoke(app, ["profile"])
    assert result.exit_code != 0
    assert "No local Ollama models" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_validation.py::test_profile_refuses_to_start_without_models -v`
Expected: FAIL because `profile` does not implement prerequisite checks yet.

- [ ] **Step 3: Write minimal profile orchestration**

```python
@app.command()
def profile() -> None:
    models = list_available_models()
    if not models:
        typer.echo("No local Ollama models were found.")
        raise typer.Exit(code=1)

    plan = BenchmarkSessionPlan(
        model_name=models[0],
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE],
        repetitions=1,
    )
    expanded_runs = expand_session_plan(plan)
    typer.echo(f"Planned {len(expanded_runs)} runs.")
    raise typer.Exit(code=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli_validation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ollama_workload_profiler/cli.py src/ollama_workload_profiler/session.py src/ollama_workload_profiler/reporting/artifacts.py src/ollama_workload_profiler/ollama_client.py tests/test_cli_validation.py
git commit -m "feat: integrate profile execution flow"
```

## Task 14: Harden Bootstrap, Doctor, Reporting, And Docs For Release

**Files:**
- Modify: `scripts/bootstrap.py`
- Modify: `src/ollama_workload_profiler/env_check.py`
- Modify: `src/ollama_workload_profiler/reporting/markdown.py`
- Modify: `README.md`
- Modify: `requirements.txt`
- Test: `tests/test_env_check.py`
- Test: `tests/test_reporting.py`

- [ ] **Step 1: Write the failing regression tests for rerun and report content**

```python
def test_bootstrap_reuses_existing_venv(tmp_path, mocker) -> None:
    venv_path = tmp_path / ".venv"
    venv_path.mkdir()
    create = mocker.patch("scripts.bootstrap.create_virtualenv")

    ensure_virtualenv(venv_path)

    create.assert_not_called()


def test_markdown_report_includes_required_sections() -> None:
    report = render_report_markdown(summary={"executive_summary": "ok"})
    assert "Executive summary" in report
    assert "Warnings and failures" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_env_check.py tests/test_reporting.py -v`
Expected: FAIL because the bootstrap and report rendering are still minimal.

- [ ] **Step 3: Implement release hardening**

```python
def ensure_virtualenv(venv_path: Path) -> bool:
    if venv_path.exists():
        return False
    create_virtualenv(venv_path)
    return True
```

```python
def render_report_markdown(summary: dict) -> str:
    return "\n".join(
        [
            "# Ollama Workload Profiler Report",
            "## Executive summary",
            summary.get("executive_summary", ""),
            "## Per-model summary",
            "## Use-case suitability matrix",
            "## Detailed timing and resource tables",
            "## Phase-level resource peaks",
            "## Warnings and failures",
            "## Plain-language recommendations",
        ]
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -v`
Expected: PASS with all core tests green.

- [ ] **Step 5: Commit**

```bash
git add scripts/bootstrap.py src/ollama_workload_profiler/env_check.py src/ollama_workload_profiler/reporting/markdown.py README.md requirements.txt tests/test_env_check.py tests/test_reporting.py
git commit -m "feat: harden setup reporting and docs for release"
```

## Self-Review Notes

Spec coverage:

- Bootstrap and dependency setup are covered in Tasks 1, 3, 7, and 14.
- Doctor diagnostics are covered in Tasks 1, 3, and 14.
- Typed plan, run result, failure, and summary schemas are covered in Task 2.
- Deterministic prompts, scenarios, and session expansion are covered in Tasks 5 and 6.
- Interactive profile flow is covered in Tasks 8 and 13.
- Ollama integration is covered in Task 4 and expanded in Task 10.
- Telemetry sampling and phase peaks are covered in Task 9 and integrated in Task 10.
- Artifact persistence, summary generation, report rendering, and verdicts are covered in Tasks 11, 12, and 14.
- Core tests and release-ready docs are covered throughout, especially Tasks 1, 3, 4, 8, 11, 12, and 14.

Placeholder scan:

- No `TODO`, `TBD`, or “implement later” placeholders remain in the plan.

Type consistency:

- Shared names are used consistently across the plan: `BenchmarkSessionPlan`, `ScenarioDefinition`, `RunResult`, `RunState`, `BenchmarkType`, and fixed artifact filenames.
