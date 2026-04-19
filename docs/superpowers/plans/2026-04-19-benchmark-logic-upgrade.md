# Benchmark Logic Upgrade Implementation Plan

> **Status:** P0 benchmark-correctness work is complete. It was merged to `main` in PR #1 and shipped/tagged as `v0.2.0`.
>
> This document is retained as the historical implementation plan and TDD execution record. The unchecked task boxes below are not current pending work; they reflect the original step-by-step plan structure.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the benchmark runtime so P0 runs are reproducible, cold and warm labels are enforced, context-fill is token-targeted, metadata is publication-ready, and reports use trustworthy aggregate metrics.

**Architecture:** Keep the existing architecture centered on `src/ollama_workload_profiler/session.py`. Extend the typed plan, dispatcher, Ollama client, prompt calibration flow, environment capture, and reporting layers in place rather than introducing a new orchestration subsystem. Persist both requested policy and effective per-run behavior in append-only artifacts so strict aggregates can exclude runs with failed enforcement.

**Tech Stack:** Python 3.13, Typer, Pydantic, httpx, psutil, pytest

---

## Completion Summary

Completed P0 outcomes on `main`:

- Benchmark execution settings are first-class plan/artifact data.
- Deterministic request settings are applied to measured Ollama calls.
- Cold-start, warm-start, and per-context warmup behavior is explicitly enforced and recorded.
- Context-fill scenarios use bounded token-targeted calibration with session-local caching.
- Raw artifacts record requested/effective settings, prep metadata, calibration status, and strict aggregate eligibility.
- Environment snapshots include best-effort Ollama, model, host, and accelerator metadata.
- Reports and summaries emphasize median/p95 statistics with sample sizes and eligibility-aware aggregates.
- README benchmark semantics were updated for the shipped behavior.

Remaining out-of-scope items are tracked by the design boundaries, not by the unchecked plan steps below.

### Task 1: Make Execution Settings First-Class And CLI-Driven

**Files:**
- Modify: `src/ollama_workload_profiler/models/plan.py`
- Modify: `src/ollama_workload_profiler/cli.py`
- Modify: `src/ollama_workload_profiler/session.py`
- Test: `tests/test_cli_validation.py`
- Test: `tests/test_session.py`

- [ ] **Step 1: Write the failing CLI and plan tests**

```python
def test_build_profile_session_plan_persists_requested_execution_settings() -> None:
    plan = build_profile_session_plan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE],
        execution_settings={
            "seed": 42,
            "temperature": 0.0,
            "top_p": None,
            "repetitions": 3,
            "warmup_runs": 1,
            "warmup_enabled": True,
        },
    )

    assert plan.execution_settings == {
        "seed": 42,
        "temperature": 0.0,
        "top_p": None,
        "repetitions": 3,
        "warmup_runs": 1,
        "warmup_enabled": True,
    }


def test_profile_help_includes_benchmark_policy_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["profile", "--help"])

    assert result.exit_code == 0
    assert "--seed" in result.stdout
    assert "--temperature" in result.stdout
    assert "--top-p" in result.stdout
    assert "--repetitions" in result.stdout
    assert "--warmup-runs" in result.stdout
    assert "--no-warmup" in result.stdout


def test_profile_command_echoes_requested_benchmark_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeClient:
        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def list_models(self) -> list[str]:
            return ["llama3.2"]

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            return {
                "response": "ok",
                "total_duration": 1_000_000,
                "load_duration": 200_000,
                "eval_count": 4,
                "eval_duration": 250_000,
                "prompt_eval_count": 8,
                "prompt_eval_duration": 500_000,
            }

    monkeypatch.setattr("ollama_workload_profiler.cli.OllamaClient", FakeClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "--model",
            "llama3.2",
            "--contexts",
            "4096",
            "--benchmark-types",
            "smoke",
            "--seed",
            "42",
            "--temperature",
            "0",
            "--repetitions",
            "3",
            "--warmup-runs",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
        input="y\n",
    )

    assert result.exit_code == 0
    assert "seed=42" in result.stdout
    assert "temperature=0.0" in result.stdout
    assert "repetitions=3" in result.stdout
    assert "warmup_runs=1" in result.stdout
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `python -m pytest tests/test_cli_validation.py -k "benchmark_policy or persists_requested_execution_settings or profile_help_includes_benchmark_policy_flags" -v`

Expected: FAIL with missing `--seed` or `execution_settings` support in the current plan builder.

- [ ] **Step 3: Move run-policy knobs into `execution_settings` and wire the CLI**

```python
class BenchmarkSessionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    contexts: list[int] = Field(min_length=1)
    benchmark_types: list[BenchmarkType] = Field(min_length=1)
    stop_conditions: list[dict[str, Any]] = Field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None
    execution_settings: dict[str, Any] = Field(default_factory=dict)


def build_profile_session_plan(
    *,
    model_name: str,
    contexts: list[int],
    benchmark_types: list[BenchmarkType],
    execution_settings: dict[str, Any] | None = None,
) -> BenchmarkSessionPlan:
    policy = {
        "seed": 42,
        "temperature": 0.0,
        "top_p": None,
        "repetitions": 1,
        "warmup_runs": 1,
        "warmup_enabled": True,
    }
    if execution_settings:
        policy.update(execution_settings)
    return BenchmarkSessionPlan(
        model_name=model_name,
        contexts=contexts,
        benchmark_types=benchmark_types,
        execution_settings=policy,
    )
```

```python
def profile(
    ...,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    temperature: Annotated[float, typer.Option("--temperature")] = 0.0,
    top_p: Annotated[float | None, typer.Option("--top-p")] = None,
    repetitions: Annotated[int, typer.Option("--repetitions", min=1)] = 1,
    warmup_runs: Annotated[int, typer.Option("--warmup-runs", min=0)] = 1,
    no_warmup: Annotated[bool, typer.Option("--no-warmup")] = False,
) -> None:
    execution_settings = {
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
        "repetitions": repetitions,
        "warmup_runs": warmup_runs,
        "warmup_enabled": not no_warmup,
    }
    plan = build_profile_session_plan(
        model_name=selected_model,
        contexts=parsed_contexts,
        benchmark_types=parsed_benchmark_types,
        execution_settings=execution_settings,
    )
```

- [ ] **Step 4: Make plan expansion and budget summaries read `repetitions` from `execution_settings`**

```python
def _plan_repetitions(plan: BenchmarkSessionPlan) -> int:
    repetitions = plan.execution_settings.get("repetitions", 1)
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("execution_settings.repetitions must be a positive integer")
    return repetitions


for repetition_index in range(1, _plan_repetitions(plan) + 1):
    ...


return {
    "context_count": len(plan.contexts),
    "benchmark_type_count": len(plan.benchmark_types),
    "scenario_count": len(scenario_keys),
    "run_count": run_count,
    "repetitions": _plan_repetitions(plan),
    "requested_execution_settings": deepcopy(plan.execution_settings),
    "warning": _budget_warning(run_count),
}
```

- [ ] **Step 5: Run the targeted tests to verify they pass**

Run: `python -m pytest tests/test_cli_validation.py tests/test_session.py -k "benchmark_policy or persists_requested_execution_settings or execution_settings" -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_cli_validation.py tests/test_session.py src/ollama_workload_profiler/models/plan.py src/ollama_workload_profiler/cli.py src/ollama_workload_profiler/session.py
git commit -m "feat: add explicit benchmark execution settings"
```

### Task 2: Add Deterministic Dispatcher Settings And Canonical Metrics

**Files:**
- Modify: `src/ollama_workload_profiler/session.py`
- Test: `tests/test_session.py`

- [ ] **Step 1: Write failing dispatcher and metrics tests**

```python
def test_ollama_dispatcher_applies_requested_generation_settings() -> None:
    planned_run = PlannedRun(
        run_id="run-settings",
        run_index=1,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.SMOKE,
        benchmark_type_index=1,
        scenario_id="smoke-basic-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="Smoke sanity check",
        scenario_version="v1",
    )
    scenario = build_benchmark_scenarios(BenchmarkType.SMOKE, 4096)[0]

    class FakeClient:
        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            assert options["seed"] == 42
            assert options["temperature"] == 0.0
            assert options["top_p"] == 0.95
            return {
                "response": "ok",
                "total_duration": 5_000_000,
                "load_duration": 1_000_000,
                "prompt_eval_count": 20,
                "prompt_eval_duration": 2_000_000_000,
                "eval_count": 10,
                "eval_duration": 1_000_000_000,
            }

    dispatcher = _OllamaDispatcher(FakeClient())
    dispatcher.execute(
        ExecutionRequest(
            run=planned_run,
            scenario=scenario,
            execution_mode=ExecutionMode.GENERATE,
            execution_settings={"seed": 42, "temperature": 0.0, "top_p": 0.95},
        )
    )


def test_response_metrics_exposes_canonical_fields() -> None:
    metrics = _response_metrics(
        {
            "load_duration": 1_500_000,
            "prompt_eval_count": 30,
            "prompt_eval_duration": 3_000_000_000,
            "eval_count": 12,
            "eval_duration": 2_000_000_000,
        }
    )

    assert metrics["load_duration"] == 1_500_000
    assert metrics["load_duration_ms"] == 1.5
    assert metrics["prompt_tokens_per_second"] == 10.0
    assert metrics["generation_tokens_per_second"] == 6.0
    assert metrics["tokens_per_second"] == metrics["generation_tokens_per_second"]
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `python -m pytest tests/test_session.py -k "applies_requested_generation_settings or exposes_canonical_fields" -v`

Expected: FAIL because the dispatcher currently sends only `num_ctx` and `num_predict`, and `_response_metrics()` does not expose load or prompt TPS fields.

- [ ] **Step 3: Extend `ExecutionRequest` plumbing and `_build_options()`**

```python
def _build_options(self, request: ExecutionRequest) -> dict[str, Any]:
    settings = dict(request.execution_settings)
    options: dict[str, Any] = {
        "num_ctx": request.run.context_size,
        "num_predict": request.scenario.target_output_tokens,
        "seed": settings["seed"],
        "temperature": settings["temperature"],
    }
    if settings.get("top_p") is not None:
        options["top_p"] = settings["top_p"]
    if request.execution_mode is ExecutionMode.TTFT:
        options["num_predict"] = max(8, request.scenario.target_output_tokens)
    return options
```

- [ ] **Step 4: Expand `_response_metrics()` to the canonical metric contract**

```python
def _response_metrics(response: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in (
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    ):
        value = response.get(key)
        if isinstance(value, (int, float)):
            metrics[key] = value

    load_duration = response.get("load_duration")
    if isinstance(load_duration, (int, float)):
        metrics["load_duration_ms"] = round(float(load_duration) / 1_000_000, 3)

    prompt_eval_count = response.get("prompt_eval_count")
    prompt_eval_duration = response.get("prompt_eval_duration")
    if isinstance(prompt_eval_count, (int, float)) and isinstance(prompt_eval_duration, (int, float)) and prompt_eval_duration:
        metrics["prompt_tokens_per_second"] = round(
            float(prompt_eval_count) / (float(prompt_eval_duration) / 1_000_000_000),
            3,
        )

    eval_count = response.get("eval_count")
    eval_duration = response.get("eval_duration")
    if isinstance(eval_count, (int, float)) and isinstance(eval_duration, (int, float)) and eval_duration:
        generation_tps = round(float(eval_count) / (float(eval_duration) / 1_000_000_000), 3)
        metrics["generation_tokens_per_second"] = generation_tps
        metrics["tokens_per_second"] = generation_tps

    return metrics
```

- [ ] **Step 5: Run the targeted tests to verify they pass**

Run: `python -m pytest tests/test_session.py -k "applies_requested_generation_settings or exposes_canonical_fields or ttft" -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_session.py src/ollama_workload_profiler/session.py src/ollama_workload_profiler/benchmarks/base.py
git commit -m "feat: add deterministic request options and canonical metrics"
```

### Task 3: Enforce Cold, Warm, And Per-Context Warmup Behavior

**Files:**
- Modify: `src/ollama_workload_profiler/ollama_client.py`
- Modify: `src/ollama_workload_profiler/session.py`
- Test: `tests/test_session.py`
- Test: `tests/test_ollama_client.py`

- [ ] **Step 1: Write failing orchestration tests for cold, warm, and warmup**

```python
def test_run_profile_session_applies_warmup_once_per_model_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = build_profile_session_plan(
        model_name="llama3.2",
        contexts=[4096, 8192],
        benchmark_types=[BenchmarkType.SMOKE],
        execution_settings={
            "seed": 42,
            "temperature": 0.0,
            "top_p": None,
            "repetitions": 1,
            "warmup_runs": 1,
            "warmup_enabled": True,
        },
    )
    warmup_calls: list[tuple[str, int]] = []

    class FakeClient:
        def list_models(self) -> list[str]:
            return ["llama3.2"]

        def warmup_model(self, *, model: str, context_size: int, options: dict[str, object]) -> None:
            warmup_calls.append((model, context_size))

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            return {"response": "ok", "total_duration": 1_000_000}

    run_profile_session(plan=plan, client=FakeClient(), output_dir=tmp_path)

    assert warmup_calls == [("llama3.2", 4096), ("llama3.2", 8192)]


def test_run_profile_session_records_failed_cold_enforcement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = build_profile_session_plan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.COLD_WARM],
        execution_settings={"seed": 42, "temperature": 0.0, "top_p": None, "repetitions": 1, "warmup_runs": 1, "warmup_enabled": True},
    )

    class FakeClient:
        def list_models(self) -> list[str]:
            return ["llama3.2"]

        def unload_model(self, *, model: str) -> bool:
            return False

        def preload_model(self, *, model: str, context_size: int, options: dict[str, object]) -> bool:
            return True

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            return {"response": "ok", "total_duration": 1_000_000}

    result = run_profile_session(plan=plan, client=FakeClient(), output_dir=tmp_path)
    cold_run = next(run for run in result.runs if run.scenario_id == "cold-warm-cold-start-v1")

    assert cold_run.metrics["requested_prep_behavior"] == "cold_start"
    assert cold_run.metrics["prep_enforcement_succeeded"] is False
    assert cold_run.metrics["eligible_for_cold_start_aggregate"] is False
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `python -m pytest tests/test_session.py tests/test_ollama_client.py -k "warmup_once_per_model_context or failed_cold_enforcement or unload_model or preload_model" -v`

Expected: FAIL because the client has no unload or preload helpers and the session loop has no prep orchestration.

- [ ] **Step 3: Add small Ollama client helpers for unload, preload, version, and show**

```python
def unload_model(self, *, model: str) -> bool:
    response = self._client.post("/api/generate", json={"model": model, "keep_alive": 0})
    response.raise_for_status()
    return True


def preload_model(self, *, model: str, context_size: int, options: Mapping[str, Any] | None = None) -> bool:
    payload_options = {"num_ctx": context_size}
    if options:
        payload_options.update(dict(options))
    self.generate(
        model=model,
        prompt="warmup",
        options={**payload_options, "num_predict": 1},
    )
    return True
```

- [ ] **Step 4: Orchestrate prep behavior and per-context warmup in `run_profile_session()`**

```python
prepared_contexts: set[tuple[str, int]] = set()

for planned_run in planned_runs:
    context_key = (planned_run.model_name, planned_run.context_size)
    prep_metrics = _prepare_run_environment(
        client=client,
        planned_run=planned_run,
        execution_settings=plan.execution_settings,
        prepared_contexts=prepared_contexts,
    )
    run_result = runner.run(planned_run)
    run_result.metrics.update(prep_metrics)
```

```python
def _prepare_run_environment(... ) -> dict[str, Any]:
    if planned_run.benchmark_type is BenchmarkType.COLD_WARM and planned_run.scenario_id.endswith("cold-start-v1"):
        succeeded = client.unload_model(model=planned_run.model_name)
        return {
            "requested_prep_behavior": "cold_start",
            "actual_prep_method": "explicit_unload",
            "prep_enforcement_succeeded": succeeded,
            "eligible_for_cold_start_aggregate": bool(succeeded),
        }

    if planned_run.benchmark_type is BenchmarkType.COLD_WARM and planned_run.scenario_id.endswith("warm-start-v1"):
        succeeded = client.preload_model(
            model=planned_run.model_name,
            context_size=planned_run.context_size,
            options={"seed": plan.execution_settings["seed"], "temperature": plan.execution_settings["temperature"]},
        )
        return {
            "requested_prep_behavior": "warm_start",
            "actual_prep_method": "explicit_preload",
            "prep_enforcement_succeeded": succeeded,
        }

    if plan.execution_settings.get("warmup_enabled", True) and context_key not in prepared_contexts:
        for _ in range(plan.execution_settings.get("warmup_runs", 1)):
            client.preload_model(...)
        prepared_contexts.add(context_key)
    return {"requested_prep_behavior": "default", "actual_prep_method": "session_warmup"}
```

- [ ] **Step 5: Run the targeted tests to verify they pass**

Run: `python -m pytest tests/test_session.py tests/test_ollama_client.py -k "warmup_once_per_model_context or failed_cold_enforcement or cold_warm or unload_model or preload_model" -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_session.py tests/test_ollama_client.py src/ollama_workload_profiler/ollama_client.py src/ollama_workload_profiler/session.py
git commit -m "feat: enforce cold warm and session warmup behavior"
```

### Task 4: Replace Char-Based Context Fill With Bounded Calibration And Caching

**Files:**
- Modify: `src/ollama_workload_profiler/prompts/scenarios.py`
- Modify: `src/ollama_workload_profiler/session.py`
- Test: `tests/test_session.py`

- [ ] **Step 1: Write failing calibration tests**

```python
def test_context_scaling_runs_record_requested_and_actual_prompt_tokens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = build_profile_session_plan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.CONTEXT_SCALING],
        execution_settings={"seed": 42, "temperature": 0.0, "top_p": None, "repetitions": 1, "warmup_runs": 1, "warmup_enabled": False},
    )
    observed_prompts: list[str] = []

    class FakeClient:
        def list_models(self) -> list[str]:
            return ["llama3.2"]

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            observed_prompts.append(prompt)
            prompt_tokens = 1000 if len(observed_prompts) == 1 else 1024
            return {
                "response": "ok",
                "total_duration": 1_000_000,
                "prompt_eval_count": prompt_tokens,
                "prompt_eval_duration": 1_000_000_000,
                "eval_count": 4,
                "eval_duration": 500_000_000,
            }

    result = run_profile_session(plan=plan, client=FakeClient(), output_dir=tmp_path)
    calibrated_run = result.runs[0]

    assert calibrated_run.metrics["requested_fill_ratio"] == 0.25
    assert calibrated_run.metrics["target_prompt_tokens"] == 1024
    assert calibrated_run.metrics["actual_prompt_tokens"] == 1024
    assert calibrated_run.metrics["calibration_status"] == "exact"


def test_context_calibration_uses_cache_key_once_per_model_context_and_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache: dict[tuple[str, int, str, str], dict[str, object]] = {}
    key = ("llama3.2", 4096, "context-scaling-low-v1", "v1")

    payload = _get_or_calibrate_context_prompt(
        calibration_cache=cache,
        cache_key=key,
        target_prompt_tokens=1024,
    )

    again = _get_or_calibrate_context_prompt(
        calibration_cache=cache,
        cache_key=key,
        target_prompt_tokens=1024,
    )

    assert again is payload
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `python -m pytest tests/test_session.py -k "calibration or actual_prompt_tokens or context_scaling_runs_record_requested" -v`

Expected: FAIL because context-fill is currently char-count based and there is no calibration cache.

- [ ] **Step 3: Add calibration metadata to scenarios**

```python
@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    ...
    fill_ratio: float | None = None
    prompt_template_version: str | None = None
    target_prompt_tokens: int | None = None
```

```python
ScenarioDefinition(
    scenario_id="context-scaling-low-v1",
    ...,
    fill_ratio=0.25,
    prompt_template_version="v1",
    target_prompt_tokens=max(1, int(context_size * 0.25)),
)
```

- [ ] **Step 4: Implement bounded calibration helpers in `session.py`**

```python
CALIBRATION_MAX_ATTEMPTS = 4
CALIBRATION_TOLERANCE_RATIO = 0.05


def _calibration_cache_key(run: PlannedRun, scenario: ScenarioDefinition) -> tuple[str, int, str, str]:
    return (
        run.model_name,
        run.context_size,
        scenario.scenario_id,
        scenario.prompt_template_version or "v1",
    )


def _calibrate_context_prompt(...) -> dict[str, Any]:
    attempt = 0
    candidate_length = scenario.target_prompt_tokens or run.context_size
    while attempt < CALIBRATION_MAX_ATTEMPTS:
        prompt = _repeat_to_length(PROMPT_SCALING_BASE_TEXT, candidate_length)
        response = client.generate(model=run.model_name, prompt=prompt, options={**options, "num_predict": 1})
        actual_prompt_tokens = int(response["prompt_eval_count"])
        if _within_tolerance(actual_prompt_tokens, scenario.target_prompt_tokens):
            return {
                "prompt": prompt,
                "actual_prompt_tokens": actual_prompt_tokens,
                "calibration_status": "exact",
                "calibration_attempts": attempt + 1,
            }
        candidate_length = _next_candidate_length(candidate_length, actual_prompt_tokens, scenario.target_prompt_tokens)
        attempt += 1
    return {
        "prompt": prompt,
        "actual_prompt_tokens": actual_prompt_tokens,
        "calibration_status": "approximate",
        "calibration_attempts": attempt,
    }
```

- [ ] **Step 5: Run the targeted tests to verify they pass**

Run: `python -m pytest tests/test_session.py -k "calibration or actual_prompt_tokens or context_scaling_runs_record_requested" -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_session.py src/ollama_workload_profiler/prompts/scenarios.py src/ollama_workload_profiler/session.py
git commit -m "feat: calibrate context fill prompts by observed tokens"
```

### Task 5: Capture Publication-Grade Environment And Model Metadata

**Files:**
- Modify: `src/ollama_workload_profiler/env_check.py`
- Modify: `src/ollama_workload_profiler/ollama_client.py`
- Modify: `src/ollama_workload_profiler/session.py`
- Test: `tests/test_env_check.py`
- Test: `tests/test_ollama_client.py`
- Test: `tests/test_session.py`

- [ ] **Step 1: Write failing metadata tests**

```python
def test_detect_accelerator_metadata_prefers_known_vendor_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "C:/Windows/System32/nvidia-smi.exe" if name == "nvidia-smi" else None)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="NVIDIA GeForce RTX 4090, 24564 MiB", stderr=""),
    )

    payload = detect_accelerator_metadata()

    assert payload["accelerator"]["vendor"] == "nvidia"
    assert payload["accelerator"]["name"] == "NVIDIA GeForce RTX 4090"
    assert payload["accelerator"]["vram_mb"] == 24564


def test_build_environment_snapshot_includes_ollama_and_model_metadata(tmp_path: Path) -> None:
    class FakeClient:
        def list_models(self) -> list[str]:
            return ["llama3.2"]

        def get_version(self) -> str:
            return "0.7.0"

        def show_model(self, *, model: str) -> dict[str, object]:
            return {"details": {"quantization_level": "Q4_K_M"}, "model_info": {"context_length": 8192}}

    environment = _build_environment_snapshot(
        plan=build_profile_session_plan(
            model_name="llama3.2",
            contexts=[4096],
            benchmark_types=[BenchmarkType.SMOKE],
            execution_settings={"seed": 42, "temperature": 0.0, "top_p": None, "repetitions": 1, "warmup_runs": 1, "warmup_enabled": True},
        ),
        available_models=["llama3.2"],
        session_timestamp=datetime(2026, 4, 19, 10, 0, tzinfo=timezone.utc),
        client=FakeClient(),
    )

    assert environment["ollama"]["version"] == "0.7.0"
    assert environment["model"]["details"]["quantization_level"] == "Q4_K_M"
    assert environment["benchmark_settings"]["seed"] == 42
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `python -m pytest tests/test_env_check.py tests/test_ollama_client.py tests/test_session.py -k "accelerator_metadata or includes_ollama_and_model_metadata or show_model or get_version" -v`

Expected: FAIL because these metadata helpers do not exist yet.

- [ ] **Step 3: Add best-effort metadata helpers**

```python
def detect_accelerator_metadata() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            name, memory = [part.strip() for part in result.stdout.split(",", 1)]
            return {"accelerator": {"vendor": "nvidia", "name": name, "vram_mb": _parse_vram_mb(memory)}}
    return {"accelerator": {"vendor": "unknown", "name": None, "vram_mb": None}}
```

```python
def get_version(self) -> str | None:
    response = self._client.get("/api/version")
    response.raise_for_status()
    payload = response.json()
    return payload.get("version")


def show_model(self, *, model: str) -> dict[str, Any]:
    response = self._client.post("/api/show", json={"model": model})
    response.raise_for_status()
    return response.json()
```

- [ ] **Step 4: Merge metadata into the environment snapshot**

```python
def _build_environment_snapshot(..., client: OllamaClient) -> dict[str, Any]:
    return {
        ...,
        "benchmark_settings": deepcopy(plan.execution_settings),
        "ollama": {
            "version": client.get_version(),
            "binary_found": detect_ollama_binary(),
        },
        "accelerator": detect_accelerator_metadata().get("accelerator"),
        "model": client.show_model(model=plan.model_name),
    }
```

- [ ] **Step 5: Run the targeted tests to verify they pass**

Run: `python -m pytest tests/test_env_check.py tests/test_ollama_client.py tests/test_session.py -k "accelerator_metadata or includes_ollama_and_model_metadata or show_model or get_version" -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_env_check.py tests/test_ollama_client.py tests/test_session.py src/ollama_workload_profiler/env_check.py src/ollama_workload_profiler/ollama_client.py src/ollama_workload_profiler/session.py
git commit -m "feat: capture ollama and accelerator metadata"
```

### Task 6: Upgrade Aggregate Summaries, Raw Artifacts, And Report Wording

**Files:**
- Modify: `src/ollama_workload_profiler/reporting/summary.py`
- Modify: `src/ollama_workload_profiler/reporting/markdown.py`
- Modify: `src/ollama_workload_profiler/reporting/artifacts.py`
- Modify: `README.md`
- Test: `tests/test_reporting.py`
- Test: `tests/test_readme.py`

- [ ] **Step 1: Write failing reporting tests**

```python
def test_build_report_summary_uses_sample_size_median_and_p95_for_completed_eligible_runs() -> None:
    runs = [
        RunResult(
            run_id="run-1",
            run_index=1,
            model_name="llama3.2",
            context_size=4096,
            context_index=1,
            benchmark_type="smoke",
            benchmark_type_index=1,
            scenario_id="smoke-basic-v1",
            scenario_index=1,
            state=RunState.COMPLETED,
            elapsed_ms=1000.0,
            metrics={
                "eligible_for_strict_aggregate": True,
                "prompt_tokens_per_second": 20.0,
                "generation_tokens_per_second": 10.0,
                "load_duration_ms": 100.0,
            },
        ),
        RunResult(
            run_id="run-2",
            run_index=2,
            model_name="llama3.2",
            context_size=4096,
            context_index=1,
            benchmark_type="smoke",
            benchmark_type_index=1,
            scenario_id="smoke-basic-v1",
            scenario_index=1,
            state=RunState.COMPLETED,
            elapsed_ms=2000.0,
            metrics={
                "eligible_for_strict_aggregate": False,
                "prompt_tokens_per_second": 200.0,
                "generation_tokens_per_second": 100.0,
                "load_duration_ms": 500.0,
            },
        ),
    ]

    summary = build_report_summary(plan={"model_name": "llama3.2"}, environment={}, runs=runs)
    row = summary.benchmark_summaries[0]

    assert row["sample_size"] == 1
    assert row["median_elapsed_ms"] == 1000.0
    assert row["median_generation_tokens_per_second"] == 10.0


def test_render_markdown_report_mentions_local_scope_and_cold_warm_method() -> None:
    summary = build_report_summary(plan={"model_name": "llama3.2"}, environment={}, runs=[])
    markdown = render_markdown_report(summary)

    assert "local serving and workload behavior" in markdown
    assert "cold and warm" in markdown
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `python -m pytest tests/test_reporting.py tests/test_readme.py -k "sample_size_median_and_p95 or local_scope_and_cold_warm_method" -v`

Expected: FAIL because the summary still uses averages and max TPS and the report wording does not describe the new benchmark semantics.

- [ ] **Step 3: Implement strict-aggregate selection and percentile helpers**

```python
def _eligible_completed_runs(grouped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        run
        for run in grouped
        if run.get("state") == "completed"
        and isinstance(run.get("metrics"), dict)
        and run["metrics"].get("eligible_for_strict_aggregate") is True
    ]


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.median(values), 3)


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.quantiles(values, n=100, method="inclusive")[94], 3)
```

```python
rows.append(
    {
        "model_name": key[0],
        "context_size": key[1],
        "benchmark_type": key[2],
        "scenario_id": key[3],
        "sample_size": len(eligible_runs),
        "completed_runs": len(completed_runs),
        "failed_runs": failed_runs,
        "stopped_runs": stopped_runs,
        "median_elapsed_ms": _median(elapsed_values),
        "p95_elapsed_ms": _p95(elapsed_values),
        "median_prompt_tokens_per_second": _median(prompt_tps_values),
        "p95_prompt_tokens_per_second": _p95(prompt_tps_values),
        "median_generation_tokens_per_second": _median(generation_tps_values),
        "p95_generation_tokens_per_second": _p95(generation_tps_values),
    }
)
```

- [ ] **Step 4: Update markdown and README wording**

```markdown
`ollama-workload-profiler` is a local serving and workload benchmark tool for Ollama.

It is designed for honest, repeatable local measurements on one machine. It is not a standardized model-quality evaluation suite.

Cold-start runs explicitly attempt to unload the model before measurement. Warm-start runs explicitly preload the model before measurement. Context-fill scenarios use bounded token-targeted calibration and are marked approximate when calibration does not converge exactly.
```

- [ ] **Step 5: Run the targeted tests to verify they pass**

Run: `python -m pytest tests/test_reporting.py tests/test_readme.py -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_reporting.py tests/test_readme.py README.md src/ollama_workload_profiler/reporting/summary.py src/ollama_workload_profiler/reporting/markdown.py src/ollama_workload_profiler/reporting/artifacts.py
git commit -m "feat: publish honest benchmark summaries and docs"
```

### Task 7: Final Verification Sweep

**Files:**
- Modify: `docs/superpowers/specs/2026-04-19-benchmark-logic-upgrade-design.md`
- Modify: `docs/superpowers/plans/2026-04-19-benchmark-logic-upgrade.md`
- Test: `tests/test_cli_validation.py`
- Test: `tests/test_env_check.py`
- Test: `tests/test_ollama_client.py`
- Test: `tests/test_reporting.py`
- Test: `tests/test_session.py`
- Test: `tests/test_readme.py`

- [ ] **Step 1: Run the full targeted benchmark-upgrade test suite**

Run: `python -m pytest tests/test_cli_validation.py tests/test_env_check.py tests/test_ollama_client.py tests/test_reporting.py tests/test_session.py tests/test_readme.py -v`

Expected: PASS

- [ ] **Step 2: Run the repository test suite**

Run: `python -m pytest -q`

Expected: PASS

- [ ] **Step 3: Review artifact contracts manually**

Run: `python -m pytest tests/test_session.py -k "artifacts or environment or calibration or cold_warm" -v`

Expected: PASS and confirms requested versus effective settings, eligibility flags, calibration status, and metadata fields are persisted in the raw and environment artifacts.

- [ ] **Step 4: Commit final cleanup if needed**

```bash
git add docs/superpowers/specs/2026-04-19-benchmark-logic-upgrade-design.md docs/superpowers/plans/2026-04-19-benchmark-logic-upgrade.md
git commit -m "chore: finalize benchmark upgrade plan artifacts"
```
