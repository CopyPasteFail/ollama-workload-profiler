import json
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
import threading
import time

import pytest
from pydantic import ValidationError

from ollama_workload_profiler.models.plan import BenchmarkSessionPlan, BenchmarkType
from ollama_workload_profiler.models.plan import PlannedRun
from ollama_workload_profiler.methodology import BENCHMARK_METHODOLOGY_VERSION
from ollama_workload_profiler.models.summary import ModelSummary, ReportSummary
from ollama_workload_profiler.models.verdicts import Verdict, VerdictLabel
from ollama_workload_profiler.models.results import RunResult, RunState
from ollama_workload_profiler.benchmarks import (
    BenchmarkExecutionStopped,
    BenchmarkRunner,
    ExecutionMode,
    ExecutionRequest,
    ExecutionResult,
    build_scenarios_for_benchmark as build_benchmark_scenarios,
    resolve_benchmark_family,
)
from ollama_workload_profiler.metrics.process import find_ollama_processes
from ollama_workload_profiler.metrics.gpu import ExternalGpuTelemetryCollector
from ollama_workload_profiler.metrics.sampler import (
    PollingProcessSampler,
    SamplePoint,
    _is_missing_gpu_backend_sample,
)
from ollama_workload_profiler.metrics.phases import compute_phase_peaks
from ollama_workload_profiler.prompts.fixtures import (
    MULTI_TURN_CHAT_TURNS,
    PROMPT_SCALING_BASE_TEXT,
    STATIC_CODE_SAMPLE,
    STATIC_SUMMARY_TEXT,
)
from ollama_workload_profiler.prompts.scenarios import (
    MultiTurnChatPromptPayload,
    TextPromptPayload,
    build_scenarios_for_benchmark,
)
from ollama_workload_profiler.session import (
    TerminalProgressReporter,
    _OllamaDispatcher,
    _build_concurrency_aggregate_metrics,
    _build_run_system_snapshot,
    _response_metrics,
    _requested_repetitions,
    build_profile_session_plan,
    expand_session_plan,
    run_profile_session,
    summarize_session_budget,
)


def _build_session_plan(
    *,
    model_name: str,
    contexts: list[int],
    benchmark_types: list[BenchmarkType],
    repetitions: int = 1,
) -> BenchmarkSessionPlan:
    return BenchmarkSessionPlan(
        model_name=model_name,
        contexts=contexts,
        benchmark_types=benchmark_types,
        execution_settings={"repetitions": repetitions},
    )


def test_benchmark_session_plan_serializes_to_json() -> None:
    plan = _build_session_plan(
        model_name="llama3.2",
        contexts=[4096, 8192],
        benchmark_types=[BenchmarkType.SMOKE, BenchmarkType.TTFT],
        repetitions=2,
    )

    payload = plan.model_dump(mode="json")

    assert payload["model_name"] == "llama3.2"
    assert payload["contexts"] == [4096, 8192]
    assert payload["benchmark_types"] == ["smoke", "ttft"]
    assert payload["benchmark_methodology_version"] == BENCHMARK_METHODOLOGY_VERSION
    assert payload["execution_settings"]["repetitions"] == 2
    assert "repetitions" not in payload


def test_benchmark_session_plan_rejects_top_level_repetitions() -> None:
    with pytest.raises(ValidationError):
        BenchmarkSessionPlan(
            model_name="llama3.2",
            contexts=[4096],
            benchmark_types=[BenchmarkType.SMOKE],
            repetitions=2,
        )


def test_benchmark_session_plan_rejects_invalid_execution_settings_repetitions() -> None:
    with pytest.raises(ValidationError, match="execution_settings.repetitions"):
        _build_session_plan(
            model_name="llama3.2",
            contexts=[4096],
            benchmark_types=[BenchmarkType.SMOKE],
            repetitions=0,
        )


def test_benchmark_session_plan_rejects_blank_model_name() -> None:
    with pytest.raises(ValidationError):
        _build_session_plan(
            model_name=" ",
            contexts=[4096],
            benchmark_types=[BenchmarkType.SMOKE],
        )


def test_benchmark_session_plan_rejects_empty_contexts() -> None:
    with pytest.raises(ValidationError):
        _build_session_plan(
            model_name="llama3.2",
            contexts=[],
            benchmark_types=[BenchmarkType.SMOKE],
        )


def test_benchmark_session_plan_rejects_empty_benchmark_types() -> None:
    with pytest.raises(ValidationError):
        _build_session_plan(
            model_name="llama3.2",
            contexts=[4096],
            benchmark_types=[],
        )


def test_expand_session_plan_uses_fixed_benchmark_execution_order_and_is_deterministic() -> None:
    plan = _build_session_plan(
        model_name="llama3.2",
        contexts=[8192, 4096],
        benchmark_types=[BenchmarkType.CONTEXT_SCALING, BenchmarkType.SMOKE],
        repetitions=1,
    )

    first_pass = expand_session_plan(plan)
    second_pass = expand_session_plan(plan)

    assert [run.run_index for run in first_pass] == list(range(1, len(first_pass) + 1))
    assert [run.run_id for run in first_pass] == [run.run_id for run in second_pass]
    assert [
        (
            run.model_name,
            run.context_size,
            run.benchmark_type,
            run.scenario_id,
            run.repetition_index,
        )
        for run in first_pass
    ] == [
        (
            run.model_name,
            run.context_size,
            run.benchmark_type,
            run.scenario_id,
            run.repetition_index,
        )
        for run in second_pass
    ]
    assert len({run.run_id for run in first_pass}) == len(first_pass)
    assert first_pass[0].run_id != "run-0001"
    assert [benchmark_type for benchmark_type, _group in groupby(run.benchmark_type for run in first_pass)] == [
        BenchmarkType.SMOKE,
        BenchmarkType.CONTEXT_SCALING,
        BenchmarkType.SMOKE,
        BenchmarkType.CONTEXT_SCALING,
    ]


def test_expand_session_plan_assigns_repetition_ordering_metadata() -> None:
    plan = _build_session_plan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE],
        repetitions=2,
    )

    runs = expand_session_plan(plan)

    assert [run.run_index for run in runs] == [1, 2]
    assert [run.repetition_index for run in runs] == [1, 2]
    assert {run.scenario_id for run in runs} == {"smoke-basic-v1"}
    assert runs[0].run_id != runs[1].run_id


def test_summarize_session_budget_uses_execution_settings_repetitions() -> None:
    plan = _build_session_plan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE, BenchmarkType.TTFT],
        repetitions=4,
    )

    budget = summarize_session_budget(plan)

    assert budget["repetitions"] == 4
    assert budget["run_count"] == 16


def test_expand_session_plan_keeps_run_ids_unique_for_repeated_contexts() -> None:
    plan = _build_session_plan(
        model_name="llama3.2",
        contexts=[4096, 4096],
        benchmark_types=[BenchmarkType.SMOKE],
        repetitions=1,
    )

    runs = expand_session_plan(plan)

    assert [run.context_index for run in runs] == [1, 2]
    assert runs[0].run_id != runs[1].run_id
    assert len({run.run_id for run in runs}) == len(runs)


def test_expand_session_plan_keeps_run_ids_unique_when_benchmark_types_repeat_in_the_plan() -> None:
    plan = _build_session_plan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE, BenchmarkType.SMOKE],
        repetitions=1,
    )

    runs = expand_session_plan(plan)

    assert [run.benchmark_type_index for run in runs] == [1]
    assert [benchmark_type for benchmark_type, _group in groupby(run.benchmark_type for run in runs)] == [
        BenchmarkType.SMOKE,
    ]
    assert len({run.run_id for run in runs}) == len(runs)


def test_expand_session_plan_normalizes_duplicate_benchmark_types_into_fixed_order() -> None:
    plan = _build_session_plan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[
            BenchmarkType.CONTEXT_SCALING,
            BenchmarkType.SMOKE,
            BenchmarkType.CONTEXT_SCALING,
        ],
        repetitions=1,
    )

    runs = expand_session_plan(plan)

    assert [benchmark_type for benchmark_type, _group in groupby(run.benchmark_type for run in runs)] == [
        BenchmarkType.SMOKE,
        BenchmarkType.CONTEXT_SCALING,
    ]


def test_run_profile_session_uses_supplied_expanded_plan_order_and_appends_raw_rows_incrementally(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = build_profile_session_plan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE, BenchmarkType.PROMPT_SCALING],
        repetitions=1,
    )
    expanded_plan = list(reversed(expand_session_plan(plan)))
    observed_run_ids: list[str] = []

    class FakeRunner:
        def __init__(self, **_: object) -> None:
            return None

        def run(self, planned_run: PlannedRun) -> RunResult:
            session_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
            assert len(session_dirs) == 1
            session_dir = session_dirs[0]

            assert (session_dir / "plan.json").exists()
            assert (session_dir / "expanded_plan.json").exists()
            assert (session_dir / "environment.json").exists()
            assert (session_dir / "raw.jsonl").exists()
            assert (session_dir / "raw.csv").exists()

            expanded_payload = json.loads((session_dir / "expanded_plan.json").read_text(encoding="utf-8"))
            assert [item["run_id"] for item in expanded_payload] == [item.run_id for item in expanded_plan]

            raw_lines = (session_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
            assert len(raw_lines) == len(observed_run_ids)

            observed_run_ids.append(planned_run.run_id)
            return RunResult(
                run_id=planned_run.run_id,
                run_index=planned_run.run_index,
                model_name=planned_run.model_name,
                context_size=planned_run.context_size,
                context_index=planned_run.context_index,
                benchmark_type=planned_run.benchmark_type,
                benchmark_type_index=planned_run.benchmark_type_index,
                scenario_id=planned_run.scenario_id,
                scenario_index=planned_run.scenario_index,
                scenario_version=planned_run.scenario_version,
                repetition_index=planned_run.repetition_index,
                scenario_name=planned_run.scenario_name,
                state=RunState.COMPLETED,
                elapsed_ms=10.0,
                metrics={"tokens_per_second": 5.0},
            )

    class FakeClient:
        def list_models(self) -> list[str]:
            return [plan.model_name]

    monkeypatch.setattr("ollama_workload_profiler.session.BenchmarkRunner", FakeRunner)

    result = run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        expanded_plan=expanded_plan,
        session_timestamp=datetime(2026, 4, 19, 10, 0, 0, tzinfo=timezone.utc),
    )

    assert observed_run_ids == [run.run_id for run in expanded_plan]

    raw_rows = (result.session_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw_rows) == len(expanded_plan)
    assert [json.loads(row)["run_id"] for row in raw_rows] == observed_run_ids
    assert {json.loads(row)["scenario_version"] for row in raw_rows} == {"v1"}

    environment_payload = json.loads((result.session_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment_payload["session_started_at"] == "2026-04-19T10:00:00+00:00"
    assert environment_payload["execution_settings"]["repetitions"] == 1
    assert "repetitions" not in environment_payload
    assert "selected_model" not in environment_payload
    assert "contexts" not in environment_payload
    assert "benchmark_types" not in environment_payload


def test_run_profile_session_persists_requested_policy_in_environment_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE],
        execution_settings={
            "repetitions": 3,
            "seed": 1234,
            "temperature": 0.2,
            "top_p": 0.9,
            "warmup_runs": 1,
            "warmup_enabled": True,
        },
    )

    class FakeRunner:
        def __init__(self, **_: object) -> None:
            return None

        def run(self, planned_run: PlannedRun) -> RunResult:
            return RunResult(
                run_id=planned_run.run_id,
                run_index=planned_run.run_index,
                model_name=planned_run.model_name,
                context_size=planned_run.context_size,
                context_index=planned_run.context_index,
                benchmark_type=planned_run.benchmark_type,
                benchmark_type_index=planned_run.benchmark_type_index,
                scenario_id=planned_run.scenario_id,
                scenario_index=planned_run.scenario_index,
                scenario_version=planned_run.scenario_version,
                repetition_index=planned_run.repetition_index,
                scenario_name=planned_run.scenario_name,
                state=RunState.COMPLETED,
                elapsed_ms=10.0,
                metrics={"tokens_per_second": 5.0},
            )

    class FakeClient:
        def list_models(self) -> list[str]:
            return [plan.model_name]

    monkeypatch.setattr("ollama_workload_profiler.session.BenchmarkRunner", FakeRunner)

    result = run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        session_timestamp=datetime(2026, 4, 19, 10, 15, 0, tzinfo=timezone.utc),
    )

    environment_payload = json.loads((result.session_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment_payload["execution_settings"] == {
        "repetitions": 3,
        "seed": 1234,
        "temperature": 0.2,
        "top_p": 0.9,
        "warmup_runs": 1,
        "warmup_enabled": True,
    }
    assert "selected_model" not in environment_payload
    assert "contexts" not in environment_payload
    assert "benchmark_types" not in environment_payload


def test_run_profile_session_persists_accelerator_and_ollama_metadata_in_environment_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = build_profile_session_plan(
        model_name="llama3.2:latest",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE],
        repetitions=1,
    )

    class FakeRunner:
        def __init__(self, **_: object) -> None:
            return None

        def run(self, planned_run: PlannedRun) -> RunResult:
            return RunResult(
                run_id=planned_run.run_id,
                run_index=planned_run.run_index,
                model_name=planned_run.model_name,
                context_size=planned_run.context_size,
                context_index=planned_run.context_index,
                benchmark_type=planned_run.benchmark_type,
                benchmark_type_index=planned_run.benchmark_type_index,
                scenario_id=planned_run.scenario_id,
                scenario_index=planned_run.scenario_index,
                scenario_version=planned_run.scenario_version,
                repetition_index=planned_run.repetition_index,
                scenario_name=planned_run.scenario_name,
                state=RunState.COMPLETED,
                elapsed_ms=10.0,
                metrics={"tokens_per_second": 5.0},
            )

    class FakeClient:
        def list_models(self) -> list[str]:
            return [plan.model_name]

        def version(self) -> str:
            return "0.6.0"

        def show_model(self, model_name: str) -> dict[str, object]:
            assert model_name == plan.model_name
            return {
                "details": {"family": "llama", "parameter_size": "8B"},
                "model_info": {"general.architecture": "llama"},
            }

    monkeypatch.setattr("ollama_workload_profiler.session.BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(
        "ollama_workload_profiler.session.detect_accelerator_metadata",
        lambda: {
            "kind": "nvidia",
            "detection_source": "nvidia-smi",
            "available": True,
            "device_count": 1,
            "devices": [{"name": "RTX 4090", "memory_total_mb": 24564}],
            "status": "detected",
            "notes": [],
        },
    )

    result = run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        session_timestamp=datetime(2026, 4, 19, 10, 20, 0, tzinfo=timezone.utc),
    )

    environment_payload = json.loads((result.session_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment_payload["accelerator"] == {
        "kind": "nvidia",
        "detection_source": "nvidia-smi",
        "available": True,
        "device_count": 1,
        "devices": [{"name": "RTX 4090", "memory_total_mb": 24564}],
        "status": "detected",
        "notes": [],
    }
    assert environment_payload["ollama"] == {
        "binary_found": True,
        "version": {"available": True, "value": "0.6.0", "error": None},
        "selected_model": {
            "name": "llama3.2:latest",
            "show_available": True,
            "details": {"family": "llama", "parameter_size": "8B"},
            "model_info": {"general.architecture": "llama"},
            "error": None,
        },
        "available_models": ["llama3.2:latest"],
    }


def test_run_profile_session_records_missing_ollama_metadata_honestly_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = build_profile_session_plan(
        model_name="llama3.2:latest",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE],
        repetitions=1,
    )

    class FakeRunner:
        def __init__(self, **_: object) -> None:
            return None

        def run(self, planned_run: PlannedRun) -> RunResult:
            return RunResult(
                run_id=planned_run.run_id,
                run_index=planned_run.run_index,
                model_name=planned_run.model_name,
                context_size=planned_run.context_size,
                context_index=planned_run.context_index,
                benchmark_type=planned_run.benchmark_type,
                benchmark_type_index=planned_run.benchmark_type_index,
                scenario_id=planned_run.scenario_id,
                scenario_index=planned_run.scenario_index,
                scenario_version=planned_run.scenario_version,
                repetition_index=planned_run.repetition_index,
                scenario_name=planned_run.scenario_name,
                state=RunState.COMPLETED,
                elapsed_ms=10.0,
                metrics={"tokens_per_second": 5.0},
            )

    class FakeClient:
        def list_models(self) -> list[str]:
            return []

        def version(self) -> str:
            raise RuntimeError("version endpoint unavailable")

        def show_model(self, model_name: str) -> dict[str, object]:
            raise RuntimeError(f"show unavailable for {model_name}")

    monkeypatch.setattr("ollama_workload_profiler.session.BenchmarkRunner", FakeRunner)
    monkeypatch.setattr("ollama_workload_profiler.session.detect_ollama_binary", lambda: False)
    monkeypatch.setattr(
        "ollama_workload_profiler.session.detect_accelerator_metadata",
        lambda: {
            "kind": "unknown",
            "detection_source": "heuristic",
            "available": False,
            "device_count": 0,
            "devices": [],
            "status": "undetected",
            "notes": ["No supported accelerator tooling detected; host is likely CPU-only."],
        },
    )

    result = run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        session_timestamp=datetime(2026, 4, 19, 10, 25, 0, tzinfo=timezone.utc),
    )

    environment_payload = json.loads((result.session_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment_payload["ollama"] == {
        "binary_found": False,
        "version": {
            "available": False,
            "value": None,
            "error": "version endpoint unavailable",
        },
        "selected_model": {
            "name": "llama3.2:latest",
            "show_available": False,
            "details": None,
            "model_info": None,
            "error": "show unavailable for llama3.2:latest",
        },
        "available_models": [],
    }


def test_run_profile_session_artifacts_keep_p0_summary_contract_consistent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = build_profile_session_plan(
        model_name="llama3.2:latest",
        contexts=[4096, 8192],
        benchmark_types=[BenchmarkType.COLD_WARM],
        repetitions=1,
    )

    class FakeRunner:
        def __init__(self, **_: object) -> None:
            self._calls = 0

        def run(self, planned_run: PlannedRun) -> RunResult:
            self._calls += 1
            is_cold = "cold-start" in planned_run.scenario_id
            metrics = {
                "load_duration_ms": 120.0 if is_cold else 5.0,
                "prompt_tokens_per_second": 1500.0 + self._calls,
                "generation_tokens_per_second": 30.0 + self._calls,
                "tokens_per_second": 30.0 + self._calls,
                "eligible_for_strict_aggregate": not is_cold,
                "eligible_for_cold_start_aggregate": is_cold,
            }
            return RunResult(
                run_id=planned_run.run_id,
                run_index=planned_run.run_index,
                model_name=planned_run.model_name,
                context_size=planned_run.context_size,
                context_index=planned_run.context_index,
                benchmark_type=planned_run.benchmark_type,
                benchmark_type_index=planned_run.benchmark_type_index,
                scenario_id=planned_run.scenario_id,
                scenario_index=planned_run.scenario_index,
                scenario_version=planned_run.scenario_version,
                repetition_index=planned_run.repetition_index,
                scenario_name=planned_run.scenario_name,
                state=RunState.COMPLETED,
                elapsed_ms=50.0 + self._calls,
                metrics=metrics,
            )

    class FakeClient:
        def __init__(self) -> None:
            self.unload_calls: list[str] = []
            self.preload_calls: list[str] = []

        def list_models(self) -> list[str]:
            return [plan.model_name]

        def unload_model(self, *, model: str) -> dict[str, object]:
            self.unload_calls.append(model)
            return {"done": True}

        def preload_model(self, *, model: str, options: dict[str, object] | None = None) -> dict[str, object]:
            self.preload_calls.append(model)
            return {"done": True}

        def version(self) -> str:
            return "0.6.0"

        def show_model(self, model_name: str) -> dict[str, object]:
            return {"details": {"family": "llama"}, "model_info": {"general.architecture": "llama"}}

    monkeypatch.setattr("ollama_workload_profiler.session.BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(
        "ollama_workload_profiler.session.detect_accelerator_metadata",
        lambda: {
            "kind": "unknown",
            "detection_source": "heuristic",
            "available": False,
            "device_count": 0,
            "devices": [],
            "status": "undetected",
            "notes": ["No supported accelerator tooling detected; host is likely CPU-only."],
        },
    )

    client = FakeClient()
    result = run_profile_session(
        plan=plan,
        client=client,
        output_dir=tmp_path,
        session_timestamp=datetime(2026, 4, 20, 9, 0, 0, tzinfo=timezone.utc),
    )

    plan_payload = json.loads((result.session_dir / "plan.json").read_text(encoding="utf-8"))
    environment_payload = json.loads((result.session_dir / "environment.json").read_text(encoding="utf-8"))
    raw_rows = [
        json.loads(line)
        for line in (result.session_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary_payload = json.loads((result.session_dir / "summary.json").read_text(encoding="utf-8"))
    report_markdown = (result.session_dir / "report.md").read_text(encoding="utf-8")

    assert plan_payload["execution_settings"]["repetitions"] == 1
    assert environment_payload["execution_settings"]["repetitions"] == 1
    assert environment_payload["execution_settings"]["seed"] == 42
    assert environment_payload["execution_settings"]["temperature"] == 0.0
    assert environment_payload["ollama"]["version"]["value"] == "0.6.0"
    assert len(raw_rows) == 4
    assert any(row["metrics"]["requested_prep_behavior"] == "cold_start" for row in raw_rows)
    assert any(row["metrics"]["requested_prep_behavior"] == "warm_start" for row in raw_rows)
    assert all(row["metrics"]["eligible_for_strict_aggregate"] is True for row in raw_rows)
    assert sum(
        1 for row in raw_rows if row["metrics"]["eligible_for_cold_start_aggregate"] is True
    ) == 2
    assert summary_payload["session_metrics"]["generation_tokens_per_second_sample_size"] == 4
    assert all("strict_sample_size" in row for row in summary_payload["benchmark_summaries"])
    assert "generation_tokens_per_second" in report_markdown
    assert "n=" in report_markdown
    assert "\n- tokens_per_second:" not in report_markdown
    assert "| tokens_per_second |" not in report_markdown
    assert client.unload_calls == [plan.model_name, plan.model_name]
    assert client.preload_calls == [plan.model_name, plan.model_name]


def test_requested_repetitions_raises_for_malformed_policy_value() -> None:
    plan = BenchmarkSessionPlan.model_construct(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE],
        execution_settings={"repetitions": 0},
    )

    with pytest.raises(ValueError, match="execution_settings.repetitions"):
        _requested_repetitions(plan)


def test_run_profile_session_records_host_metadata_and_per_run_system_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = build_profile_session_plan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE],
        repetitions=1,
    )

    class FakeRunner:
        def __init__(self, **_: object) -> None:
            return None

        def run(self, planned_run: PlannedRun) -> RunResult:
            return RunResult(
                run_id=planned_run.run_id,
                run_index=planned_run.run_index,
                model_name=planned_run.model_name,
                context_size=planned_run.context_size,
                context_index=planned_run.context_index,
                benchmark_type=planned_run.benchmark_type,
                benchmark_type_index=planned_run.benchmark_type_index,
                scenario_id=planned_run.scenario_id,
                scenario_index=planned_run.scenario_index,
                scenario_version=planned_run.scenario_version,
                repetition_index=planned_run.repetition_index,
                scenario_name=planned_run.scenario_name,
                state=RunState.COMPLETED,
                elapsed_ms=10.0,
                metrics={"tokens_per_second": 5.0},
            )

    class FakeClient:
        def list_models(self) -> list[str]:
            return [plan.model_name]

    monkeypatch.setattr("ollama_workload_profiler.session.BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(
        "ollama_workload_profiler.session._build_host_metadata",
        lambda: {
            "hostname": "bench-host",
            "os": {"platform": "test-os", "release": "1.0"},
            "cpu": {"logical_cores": 16, "physical_cores": 8},
            "memory": {"total_mb": 32768.0},
        },
    )
    monkeypatch.setattr(
        "ollama_workload_profiler.session._build_run_system_snapshot",
        lambda: {
            "cpu_percent": 21.5,
            "system_cpu_load_snapshot": 21.5,
            "memory_available_mb": 24000.0,
            "available_system_memory_mb": 24000.0,
            "memory_used_percent": 26.8,
            "host_pressure_warning": False,
            "host_pressure_warning_reasons": [],
            "ollama_process_count": 2,
        },
    )

    result = run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        session_timestamp=datetime(2026, 4, 19, 10, 30, 0, tzinfo=timezone.utc),
    )

    environment_payload = json.loads((result.session_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment_payload["host"]["hostname"] == "bench-host"
    assert environment_payload["host"]["cpu"]["logical_cores"] == 16
    assert environment_payload["host"]["memory"]["total_mb"] == 32768.0

    raw_rows = [
        json.loads(line)
        for line in (result.session_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert raw_rows[0]["system_snapshot"] == {
        "cpu_percent": 21.5,
        "system_cpu_load_snapshot": 21.5,
        "memory_available_mb": 24000.0,
        "available_system_memory_mb": 24000.0,
        "memory_used_percent": 26.8,
        "host_pressure_warning": False,
        "host_pressure_warning_reasons": [],
        "ollama_process_count": 2,
    }


def test_build_run_system_snapshot_records_host_pressure_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeVirtualMemory:
        available = 768 * 1024 * 1024
        percent = 91.5

    monkeypatch.setattr("ollama_workload_profiler.session._safe_virtual_memory", lambda: FakeVirtualMemory())
    monkeypatch.setattr("ollama_workload_profiler.session._safe_cpu_percent", lambda: 87.5)
    monkeypatch.setattr("ollama_workload_profiler.session.find_ollama_processes", lambda: [])

    snapshot = _build_run_system_snapshot()

    assert snapshot["available_system_memory_mb"] == 768.0
    assert snapshot["system_cpu_load_snapshot"] == 87.5
    assert snapshot["host_pressure_warning"] is True
    assert snapshot["host_pressure_warning_reasons"] == [
        "system_cpu_load_snapshot >= 80%",
        "memory_used_percent >= 90%",
        "available_system_memory_mb < 1024",
    ]


def test_build_run_system_snapshot_keeps_host_pressure_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_process_discovery() -> list[object]:
        raise RuntimeError("process table unavailable")

    monkeypatch.setattr("ollama_workload_profiler.session._safe_virtual_memory", lambda: None)
    monkeypatch.setattr("ollama_workload_profiler.session._safe_cpu_percent", lambda: None)
    monkeypatch.setattr("ollama_workload_profiler.session.find_ollama_processes", fail_process_discovery)

    snapshot = _build_run_system_snapshot()

    assert snapshot["available_system_memory_mb"] is None
    assert snapshot["system_cpu_load_snapshot"] is None
    assert snapshot["host_pressure_warning"] is False
    assert snapshot["host_pressure_warning_reasons"] == []
    assert snapshot["ollama_process_count"] is None


def test_run_profile_session_persists_phase_peaks_from_real_sampler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = build_profile_session_plan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE],
        repetitions=1,
    )

    class FakeMemoryInfo:
        def __init__(self, rss: int) -> None:
            self.rss = rss

    class FakeProcess:
        def __init__(self) -> None:
            self._cpu_values = iter([5.0, 35.0, 12.0])

        def memory_info(self) -> FakeMemoryInfo:
            return FakeMemoryInfo(192 * 1024 * 1024)

        def cpu_percent(self, interval: float | None = None) -> float:
            return next(self._cpu_values, 12.0)

    class FakeClient:
        def list_models(self) -> list[str]:
            return [plan.model_name]

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            time.sleep(0.01)
            return {
                "response": "ok",
                "total_duration": 1_000_000,
                "eval_count": 4,
                "eval_duration": 250_000,
                "prompt_eval_count": 8,
                "prompt_eval_duration": 500_000,
            }

    fake_processes = [FakeProcess()]
    monkeypatch.setattr(
        "ollama_workload_profiler.session.find_ollama_processes",
        lambda: fake_processes,
    )

    result = run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        session_timestamp=datetime(2026, 4, 19, 11, 0, 0, tzinfo=timezone.utc),
    )

    raw_rows = [
        json.loads(line)
        for line in (result.session_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(raw_rows) == 1
    assert raw_rows[0]["metrics"]["phase_peaks"]["generation"]["rss_mb"] == 192.0
    assert raw_rows[0]["metrics"]["phase_peaks"]["generation"]["cpu_percent"] == 35.0


def test_run_profile_session_notifies_progress_reporter_for_each_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = build_profile_session_plan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE, BenchmarkType.TTFT],
        repetitions=1,
    )
    progress_events: list[tuple[str, object]] = []

    class FakeRunner:
        def __init__(self, **_: object) -> None:
            return None

        def run(self, planned_run: PlannedRun) -> RunResult:
            return RunResult(
                run_id=planned_run.run_id,
                run_index=planned_run.run_index,
                model_name=planned_run.model_name,
                context_size=planned_run.context_size,
                context_index=planned_run.context_index,
                benchmark_type=planned_run.benchmark_type,
                benchmark_type_index=planned_run.benchmark_type_index,
                scenario_id=planned_run.scenario_id,
                scenario_index=planned_run.scenario_index,
                scenario_version=planned_run.scenario_version,
                repetition_index=planned_run.repetition_index,
                scenario_name=planned_run.scenario_name,
                state=RunState.COMPLETED,
                elapsed_ms=10.0,
                metrics={"tokens_per_second": 5.0},
            )

    class FakeClient:
        def list_models(self) -> list[str]:
            return [plan.model_name]

    class FakeProgressReporter:
        def on_run_started(self, planned_run: PlannedRun, *, total_runs: int) -> None:
            progress_events.append(
                (
                    "start",
                    planned_run.run_index,
                    total_runs,
                    planned_run.benchmark_type.value,
                    planned_run.scenario_name,
                    planned_run.context_size,
                    planned_run.repetition_index,
                )
            )

        def on_run_finished(self, run_result: RunResult, *, total_runs: int) -> None:
            progress_events.append(
                (
                    "finish",
                    run_result.run_index,
                    total_runs,
                    run_result.state.value,
                )
            )

        def on_session_finished(self, summary: ReportSummary, *, total_runs: int) -> None:
            progress_events.append(
                (
                    "session",
                    total_runs,
                    summary.session_metrics["completed_runs"],
                    summary.session_metrics["failed_runs"],
                )
            )

    monkeypatch.setattr("ollama_workload_profiler.session.BenchmarkRunner", FakeRunner)

    run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        progress_reporter=FakeProgressReporter(),
        session_timestamp=datetime(2026, 4, 19, 11, 30, 0, tzinfo=timezone.utc),
    )

    assert progress_events == [
        ("start", 1, 4, "smoke", "Smoke sanity check", 4096, 1),
        ("finish", 1, 4, "completed"),
        ("start", 2, 4, "ttft", "TTFT probe", 4096, 1),
        ("finish", 2, 4, "completed"),
        ("start", 3, 4, "ttft", "TTFT chat probe", 4096, 1),
        ("finish", 3, 4, "completed"),
        ("start", 4, 4, "ttft", "TTFT stream-shape probe", 4096, 1),
        ("finish", 4, 4, "completed"),
        ("session", 4, 4, 0),
    ]


def test_terminal_progress_reporter_emits_default_and_live_progress_lines() -> None:
    outputs: list[tuple[str, bool]] = []
    clock_values = iter([100.0, 101.25])
    planned_run = PlannedRun(
        run_id="run-progress",
        run_index=7,
        model_name="gemma4:latest",
        context_size=32768,
        context_index=1,
        benchmark_type=BenchmarkType.CONTEXT_SCALING,
        benchmark_type_index=1,
        scenario_id="context-scaling-high-v1",
        scenario_index=3,
        repetition_index=2,
        scenario_name="High fill context",
        scenario_version="v1",
    )
    run_result = RunResult(
        run_id=planned_run.run_id,
        run_index=planned_run.run_index,
        model_name=planned_run.model_name,
        context_size=planned_run.context_size,
        context_index=planned_run.context_index,
        benchmark_type=planned_run.benchmark_type,
        benchmark_type_index=planned_run.benchmark_type_index,
        scenario_id=planned_run.scenario_id,
        scenario_index=planned_run.scenario_index,
        scenario_version=planned_run.scenario_version,
        repetition_index=planned_run.repetition_index,
        scenario_name=planned_run.scenario_name,
        state=RunState.COMPLETED,
        elapsed_ms=1250.0,
        metrics={"tokens_per_second": 42.5},
    )

    reporter = TerminalProgressReporter(
        echo=lambda message, nl=True: outputs.append((message, nl)),
        live=True,
        telemetry_provider=lambda: {"cpu_percent": 73.5, "rss_mb": 2048.0, "tokens_per_second": None},
        clock=lambda: next(clock_values),
        poll_interval_seconds=60.0,
        terminal_width_provider=lambda: 200,
    )

    reporter.on_run_started(planned_run, total_runs=42)
    reporter.on_run_finished(run_result, total_runs=42)

    lines = [message for message, _nl in outputs]
    assert any("Run 7/42" in line for line in lines)
    assert any("context-scaling" in line for line in lines)
    assert any("High fill context" in line for line in lines)
    assert any("ctx 32768" in line for line in lines)
    assert any("rep 2" in line for line in lines)
    assert any("elapsed 1s" in line for line in lines)
    assert any("CPU 73.5%" in line and "RSS 2048.0 MB" in line for line in lines)
    assert any("completed 0 | pending 42 | failed 0" in line for line in lines)
    assert any(
        "Result | run 7/42 | completed | context-scaling | High fill context" in line
        and "elapsed 1.2s" in line
        and "tok/s 42.5" in line
        for line in lines
    )


def test_terminal_progress_reporter_redraws_live_status_in_place() -> None:
    outputs: list[tuple[str, bool]] = []
    clock_values = iter([10.0, 10.5])
    planned_run = PlannedRun(
        run_id="run-live-redraw",
        run_index=2,
        model_name="gemma4:e2b",
        context_size=65536,
        context_index=1,
        benchmark_type=BenchmarkType.PROMPT_SCALING,
        benchmark_type_index=1,
        scenario_id="prompt-scaling-large-v1",
        scenario_index=3,
        repetition_index=1,
        scenario_name="Large prompt",
        scenario_version="v1",
    )
    run_result = RunResult(
        run_id=planned_run.run_id,
        run_index=planned_run.run_index,
        model_name=planned_run.model_name,
        context_size=planned_run.context_size,
        context_index=planned_run.context_index,
        benchmark_type=planned_run.benchmark_type,
        benchmark_type_index=planned_run.benchmark_type_index,
        scenario_id=planned_run.scenario_id,
        scenario_index=planned_run.scenario_index,
        scenario_version=planned_run.scenario_version,
        repetition_index=planned_run.repetition_index,
        scenario_name=planned_run.scenario_name,
        state=RunState.COMPLETED,
        elapsed_ms=500.0,
        metrics={},
    )

    reporter = TerminalProgressReporter(
        echo=lambda message, nl=True: outputs.append((message, nl)),
        live=True,
        telemetry_provider=lambda: {"cpu_percent": 50.0, "rss_mb": 1024.0, "tokens_per_second": None},
        clock=lambda: next(clock_values),
        poll_interval_seconds=60.0,
    )

    reporter.on_run_started(planned_run, total_runs=5)
    reporter.on_run_finished(run_result, total_runs=5)

    live_updates = [(message, nl) for message, nl in outputs if message.startswith("\rLive ")]
    result_rows = [(message, nl) for message, nl in outputs if "Result | run 2/5" in message]

    assert live_updates
    assert all(nl is False for _message, nl in live_updates)
    assert result_rows
    assert all(nl is True for _message, nl in result_rows)


def test_terminal_progress_reporter_truncates_live_status_to_terminal_width() -> None:
    outputs: list[tuple[str, bool]] = []
    clock_values = iter([10.0, 10.5])
    planned_run = PlannedRun(
        run_id="run-live-width",
        run_index=2,
        model_name="gemma4:e2b",
        context_size=65536,
        context_index=1,
        benchmark_type=BenchmarkType.PROMPT_SCALING,
        benchmark_type_index=1,
        scenario_id="prompt-scaling-large-v1",
        scenario_index=3,
        repetition_index=1,
        scenario_name="Large prompt",
        scenario_version="v1",
    )

    reporter = TerminalProgressReporter(
        echo=lambda message, nl=True: outputs.append((message, nl)),
        live=True,
        telemetry_provider=lambda: {"cpu_percent": 50.0, "rss_mb": 1024.0, "tokens_per_second": None},
        clock=lambda: next(clock_values),
        poll_interval_seconds=60.0,
        terminal_width_provider=lambda: 40,
    )

    reporter.on_run_started(planned_run, total_runs=5)

    live_updates = [message for message, nl in outputs if nl is False]

    assert live_updates
    assert all(message.startswith("\r") for message in live_updates)
    assert all(len(message.removeprefix("\r")) <= 40 for message in live_updates)


def test_terminal_progress_reporter_emits_live_session_summary_from_dict_metrics() -> None:
    outputs: list[tuple[str, bool]] = []
    summary = ReportSummary(
        session_metrics={
            "completed_runs": 6,
            "failed_runs": 0,
        }
    )
    reporter = TerminalProgressReporter(
        echo=lambda message, nl=True: outputs.append((message, nl)),
        live=True,
        poll_interval_seconds=60.0,
    )

    reporter.on_session_finished(summary, total_runs=6)

    assert outputs == [("\rSession summary | completed 6 | failed 0 | total 6", True)]


def test_compute_phase_peaks_groups_samples_by_phase() -> None:
    peaks = compute_phase_peaks(
        samples=[
            {"phase": "load", "rss_mb": 100, "cpu_percent": 5},
            {"phase": "generation", "rss_mb": 150, "cpu_percent": 35},
        ]
    )

    assert peaks["generation"].rss_mb == 150
    assert peaks["generation"].cpu_percent == 35
    assert peaks["load"].rss_mb == 100


def test_compute_phase_peaks_breaks_rss_ties_by_higher_cpu_percent() -> None:
    peaks = compute_phase_peaks(
        samples=[
            {"phase": "generation", "rss_mb": 150, "cpu_percent": 12},
            {"phase": "generation", "rss_mb": 150, "cpu_percent": 35},
        ]
    )

    assert peaks["generation"].rss_mb == 150
    assert peaks["generation"].cpu_percent == 35


def test_external_gpu_telemetry_collector_parses_nvidia_smi_multi_gpu_output() -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> str:
        commands.append(command)
        return "1024, 40\n2048, 80\n"

    collector = ExternalGpuTelemetryCollector(
        executable_finder=lambda name: f"C:/bin/{name}.exe" if name == "nvidia-smi" else None,
        command_runner=fake_run,
    )

    sample = collector.sample()

    assert commands == [
        [
            "C:/bin/nvidia-smi.exe",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    ]
    assert sample.available is True
    assert sample.source == "nvidia-smi"
    assert sample.memory_used_mb == 3072.0
    assert sample.util_percent == 60.0
    assert sample.device_count == 2
    assert sample.notes == ["multi_gpu_memory_summed_util_averaged"]


def test_external_gpu_telemetry_collector_reports_unavailable_without_running_command() -> None:
    collector = ExternalGpuTelemetryCollector(
        executable_finder=lambda _name: None,
        command_runner=lambda _command: "should not run",
    )

    sample = collector.sample()

    assert sample.available is False
    assert sample.source is None
    assert sample.memory_used_mb is None
    assert sample.util_percent is None
    assert sample.device_count == 0
    assert sample.notes == ["nvidia-smi not found; rocm-smi and Apple Silicon live GPU telemetry are not supported yet"]


def test_external_gpu_telemetry_collector_treats_command_failures_as_unavailable() -> None:
    def failing_run(_command: list[str]) -> str:
        raise RuntimeError("nvidia-smi failed")

    collector = ExternalGpuTelemetryCollector(
        executable_finder=lambda name: name if name == "nvidia-smi" else None,
        command_runner=failing_run,
    )

    sample = collector.sample()

    assert sample.available is False
    assert sample.source == "nvidia-smi"
    assert sample.error == "nvidia-smi failed"
    assert sample.memory_used_mb is None
    assert sample.util_percent is None


@pytest.mark.parametrize(
    "sample",
    [
        {"phase": None, "rss_mb": 100, "cpu_percent": 5},
        {"phase": True, "rss_mb": 100, "cpu_percent": 5},
        {"phase": "load", "rss_mb": float("nan"), "cpu_percent": 5},
        {"phase": "load", "rss_mb": 100, "cpu_percent": float("inf")},
        {"phase": "load", "rss_mb": True, "cpu_percent": 5},
    ],
)
def test_compute_phase_peaks_rejects_malformed_telemetry(sample: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        compute_phase_peaks([sample])


def test_find_ollama_processes_filters_process_names_case_insensitively(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        def __init__(self, name: str, pid: int, ppid: int | None = None) -> None:
            self.info = {"name": name, "pid": pid, "ppid": ppid}

    monkeypatch.setattr(
        "ollama_workload_profiler.metrics.process.psutil.process_iter",
        lambda attrs: [
            FakeProcess("ollama", 10),
            FakeProcess("Ollama-helper", 11),
            FakeProcess("python", 12),
        ],
    )

    processes = find_ollama_processes()

    assert [process.info["name"] for process in processes] == ["ollama", "Ollama-helper"]


def test_find_ollama_processes_includes_descendants_of_ollama_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        def __init__(self, name: str, pid: int, ppid: int | None = None) -> None:
            self.info = {"name": name, "pid": pid, "ppid": ppid}

    monkeypatch.setattr(
        "ollama_workload_profiler.metrics.process.psutil.process_iter",
        lambda attrs: [
            FakeProcess("ollama", 10, 1),
            FakeProcess("runner", 20, 10),
            FakeProcess("worker", 30, 20),
            FakeProcess("unrelated", 40, 1),
        ],
    )

    processes = find_ollama_processes()

    assert [process.info["pid"] for process in processes] == [10, 20, 30]


def test_find_ollama_processes_falls_back_to_roots_when_parent_metadata_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self, name: str, pid: int | None = None) -> None:
            self.info = {"name": name}
            if pid is not None:
                self.info["pid"] = pid

    monkeypatch.setattr(
        "ollama_workload_profiler.metrics.process.psutil.process_iter",
        lambda attrs: [
            FakeProcess("ollama", 10),
            FakeProcess("python"),
        ],
    )

    processes = find_ollama_processes()

    assert [process.info["name"] for process in processes] == ["ollama"]


def test_polling_process_sampler_collects_load_generation_and_post_run_samples() -> None:
    class FakeMemoryInfo:
        def __init__(self, rss: int) -> None:
            self.rss = rss

    class FakeProcess:
        def memory_info(self) -> FakeMemoryInfo:
            return FakeMemoryInfo(256 * 1024 * 1024)

        def cpu_percent(self, interval: float | None = None) -> float:
            return 12.5

    planned_run = PlannedRun(
        run_id="run-sampler",
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
    fake_process = FakeProcess()
    sampler = PollingProcessSampler(
        process_finder=lambda: [fake_process],
        interval_seconds=0.001,
        shutdown_timeout_seconds=0.5,
    )

    sampler.start(planned_run)
    time.sleep(0.01)
    samples = sampler.stop()

    assert samples[0].phase == "load"
    assert samples[-1].phase == "resident_post_run"
    assert any(sample.phase == "generation" for sample in samples)
    assert all(sample.rss_mb == 256.0 for sample in samples)
    assert all(sample.cpu_percent == 12.5 for sample in samples)


def test_polling_process_sampler_records_sampled_process_metadata() -> None:
    class FakeMemoryInfo:
        def __init__(self, rss: int) -> None:
            self.rss = rss

    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.info = {"pid": pid}

        def memory_info(self) -> FakeMemoryInfo:
            return FakeMemoryInfo(32 * 1024 * 1024)

        def cpu_percent(self, interval: float | None = None) -> float:
            return 5.0

    sampler = PollingProcessSampler(
        process_finder=lambda: [FakeProcess(10), FakeProcess(20)],
    )

    sample = sampler._snapshot_sample("generation")

    assert sample.sampled_process_count == 2
    assert sample.sampled_process_ids == [10, 20]
    assert sample.rss_mb == 64.0
    assert sample.cpu_percent == 10.0


def test_polling_process_sampler_counts_unique_sampled_process_ids() -> None:
    class FakeMemoryInfo:
        def __init__(self, rss: int) -> None:
            self.rss = rss

    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.info = {"pid": pid}

        def memory_info(self) -> FakeMemoryInfo:
            return FakeMemoryInfo(32 * 1024 * 1024)

        def cpu_percent(self, interval: float | None = None) -> float:
            return 5.0

    sampler = PollingProcessSampler(
        process_finder=lambda: [FakeProcess(10), FakeProcess(10), FakeProcess(20)],
    )

    sample = sampler._snapshot_sample("generation")

    assert sample.sampled_process_ids == [10, 20]
    assert sample.sampled_process_count == len(sample.sampled_process_ids)


def test_polling_process_sampler_includes_best_effort_gpu_samples() -> None:
    class FakeMemoryInfo:
        def __init__(self, rss: int) -> None:
            self.rss = rss

    class FakeProcess:
        def memory_info(self) -> FakeMemoryInfo:
            return FakeMemoryInfo(64 * 1024 * 1024)

        def cpu_percent(self, interval: float | None = None) -> float:
            return 10.0

    class FakeGpuCollector:
        def sample(self) -> object:
            return type(
                "GpuSample",
                (),
                {
                    "available": True,
                    "source": "nvidia-smi",
                    "memory_used_mb": 2048.0,
                    "util_percent": 55.0,
                    "device_count": 1,
                    "notes": [],
                    "error": None,
                },
            )()

    planned_run = PlannedRun(
        run_id="run-sampler-gpu",
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
    sampler = PollingProcessSampler(
        process_finder=lambda: [FakeProcess()],
        gpu_collector=FakeGpuCollector(),
        interval_seconds=0.001,
        shutdown_timeout_seconds=0.5,
    )

    sampler.start(planned_run)
    time.sleep(0.01)
    samples = sampler.stop()

    assert any(sample.gpu_telemetry_available for sample in samples)
    assert max(sample.gpu_memory_used_mb or 0 for sample in samples) == 2048.0
    assert max(sample.gpu_util_percent or 0 for sample in samples) == 55.0


def test_polling_process_sampler_reuses_cached_gpu_sample_between_gpu_poll_intervals() -> None:
    class FakeGpuCollector:
        def __init__(self) -> None:
            self.calls = 0

        def sample(self) -> object:
            self.calls += 1
            return type(
                "GpuSample",
                (),
                {
                    "available": True,
                    "source": "nvidia-smi",
                    "memory_used_mb": 1024.0 + self.calls,
                    "util_percent": 10.0 + self.calls,
                    "device_count": 1,
                    "notes": [],
                    "error": None,
                },
            )()

    now = 100.0
    collector = FakeGpuCollector()
    sampler = PollingProcessSampler(
        process_finder=lambda: [],
        gpu_collector=collector,
        interval_seconds=0.001,
        gpu_poll_interval_seconds=1.0,
        clock=lambda: now,
    )

    first = sampler._snapshot_sample("load")
    second = sampler._snapshot_sample("generation")

    assert collector.calls == 1
    assert first.gpu_memory_used_mb == 1025.0
    assert second.gpu_memory_used_mb == 1025.0


def test_polling_process_sampler_caches_missing_gpu_backend_for_the_run() -> None:
    class FakeGpuCollector:
        def __init__(self) -> None:
            self.calls = 0

        def sample(self) -> object:
            self.calls += 1
            return type(
                "GpuSample",
                (),
                {
                    "available": False,
                    "source": None,
                    "memory_used_mb": None,
                    "util_percent": None,
                    "device_count": 0,
                    "notes": ["nvidia-smi not found"],
                    "error": None,
                },
            )()

    now = 100.0
    collector = FakeGpuCollector()
    sampler = PollingProcessSampler(
        process_finder=lambda: [],
        gpu_collector=collector,
        gpu_poll_interval_seconds=0.5,
        clock=lambda: now,
    )

    first = sampler._snapshot_sample("load")
    now = 200.0
    second = sampler._snapshot_sample("generation")

    assert collector.calls == 1
    assert first.gpu_telemetry_available is False
    assert second.gpu_telemetry_available is False
    assert second.gpu_telemetry_notes == ["nvidia-smi not found"]


def test_missing_gpu_backend_cache_uses_named_internal_signal() -> None:
    assert _is_missing_gpu_backend_sample(
        type(
            "GpuSample",
            (),
            {
                "available": False,
                "source": None,
                "error": None,
            },
        )()
    )
    assert not _is_missing_gpu_backend_sample(
        type(
            "GpuSample",
            (),
            {
                "available": False,
                "source": "nvidia-smi",
                "error": "command failed",
            },
        )()
    )


def test_polling_process_sampler_retries_failed_gpu_command_only_after_gpu_poll_interval() -> None:
    class FakeGpuCollector:
        def __init__(self) -> None:
            self.calls = 0

        def sample(self) -> object:
            self.calls += 1
            return type(
                "GpuSample",
                (),
                {
                    "available": False,
                    "source": "nvidia-smi",
                    "memory_used_mb": None,
                    "util_percent": None,
                    "device_count": 0,
                    "notes": [],
                    "error": f"failed-{self.calls}",
                },
            )()

    now = 100.0
    collector = FakeGpuCollector()
    sampler = PollingProcessSampler(
        process_finder=lambda: [],
        gpu_collector=collector,
        gpu_poll_interval_seconds=0.5,
        clock=lambda: now,
    )

    first = sampler._snapshot_sample("load")
    now = 100.25
    second = sampler._snapshot_sample("generation")
    now = 100.51
    third = sampler._snapshot_sample("generation")

    assert collector.calls == 2
    assert first.gpu_telemetry_error == "failed-1"
    assert second.gpu_telemetry_error == "failed-1"
    assert third.gpu_telemetry_error == "failed-2"


def test_polling_process_sampler_waits_for_in_flight_poll_before_stop_returns() -> None:
    entered_generation = threading.Event()
    release_generation = threading.Event()

    class FakeMemoryInfo:
        def __init__(self, rss: int) -> None:
            self.rss = rss

    class FakeProcess:
        def __init__(self) -> None:
            self._cpu_calls = 0

        def memory_info(self) -> FakeMemoryInfo:
            return FakeMemoryInfo(128 * 1024 * 1024)

        def cpu_percent(self, interval: float | None = None) -> float:
            self._cpu_calls += 1
            if self._cpu_calls == 2:
                entered_generation.set()
                release_generation.wait(timeout=1.0)
            return 10.0

    planned_run = PlannedRun(
        run_id="run-sampler-stop",
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
    fake_process = FakeProcess()
    sampler = PollingProcessSampler(
        process_finder=lambda: [fake_process],
        interval_seconds=0.001,
        shutdown_timeout_seconds=0.5,
    )
    stop_completed = threading.Event()

    sampler.start(planned_run)
    assert entered_generation.wait(timeout=1.0) is True

    stop_thread = threading.Thread(
        target=lambda: (sampler.stop(), stop_completed.set()),
        daemon=True,
    )
    stop_thread.start()
    time.sleep(0.15)

    assert stop_completed.is_set() is False

    release_generation.set()
    stop_thread.join(timeout=1.0)

    assert stop_completed.is_set() is True


def test_polling_process_sampler_stop_returns_when_post_run_snapshot_stalls() -> None:
    allow_post_run = threading.Event()

    class FakeMemoryInfo:
        def __init__(self, rss: int) -> None:
            self.rss = rss

    class FakeProcess:
        def __init__(self) -> None:
            self._cpu_calls = 0

        def memory_info(self) -> FakeMemoryInfo:
            return FakeMemoryInfo(128 * 1024 * 1024)

        def cpu_percent(self, interval: float | None = None) -> float:
            self._cpu_calls += 1
            if self._cpu_calls >= 3:
                allow_post_run.wait(timeout=1.0)
            return 10.0

    planned_run = PlannedRun(
        run_id="run-sampler-post-run",
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
    fake_process = FakeProcess()
    sampler = PollingProcessSampler(
        process_finder=lambda: [fake_process],
        interval_seconds=0.001,
        shutdown_timeout_seconds=0.05,
    )
    stop_completed = threading.Event()

    sampler.start(planned_run)
    time.sleep(0.01)

    stop_thread = threading.Thread(
        target=lambda: (sampler.stop(), stop_completed.set()),
        daemon=True,
    )
    stop_thread.start()
    time.sleep(0.2)

    assert stop_completed.is_set() is True


def test_report_summary_uses_typed_model_summaries() -> None:
    summary = ReportSummary(
        model_summaries={
            "llama3.2": ModelSummary(
                summary="solid",
                metrics={"ttft_ms": 123.4},
            )
        }
    )

    payload = summary.model_dump(mode="json")

    assert payload["model_summaries"]["llama3.2"]["summary"] == "solid"
    assert payload["model_summaries"]["llama3.2"]["metrics"]["ttft_ms"] == 123.4


def test_verdict_label_is_machine_friendly_but_still_displayable() -> None:
    assert VerdictLabel.GOOD_FIT.value == "good_fit"
    assert VerdictLabel.GOOD_FIT.display_label == "Good fit"


def test_verdict_serializes_machine_friendly_label() -> None:
    verdict = Verdict(
        label=VerdictLabel.GOOD_FIT,
        rationale="Core metrics are within acceptable bands.",
        supporting_metrics={"ttft_ms": 123.4},
    )

    payload = verdict.model_dump(mode="json")

    assert payload["label"] == "good_fit"


def test_run_result_has_stable_ordering_fields() -> None:
    result = RunResult(
        run_id="run-0001",
        run_index=1,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type="smoke",
        benchmark_type_index=1,
        scenario_id="smoke-basic-v1",
        scenario_index=1,
        scenario_version="v1",
        state=RunState.PLANNED,
        repetition_index=1,
    )

    assert result.run_index == 1
    assert result.run_id == "run-0001"
    assert result.context_index == 1
    assert result.benchmark_type_index == 1
    assert result.scenario_index == 1
    assert result.scenario_version == "v1"


def test_prompt_scaling_builds_three_scenarios() -> None:
    scenarios = build_scenarios_for_benchmark(BenchmarkType.PROMPT_SCALING, 4096)
    scenarios_by_id = {item.scenario_id: item for item in scenarios}
    small_prompt = scenarios_by_id["prompt-scaling-small-v1"].prompt_payload
    medium_prompt = scenarios_by_id["prompt-scaling-medium-v1"].prompt_payload
    large_prompt = scenarios_by_id["prompt-scaling-large-v1"].prompt_payload

    assert set(scenarios_by_id) == {
        "prompt-scaling-small-v1",
        "prompt-scaling-medium-v1",
        "prompt-scaling-large-v1",
    }
    assert isinstance(small_prompt, TextPromptPayload)
    assert len(small_prompt.text) == 1024
    assert len(medium_prompt.text) == 4096
    assert len(large_prompt.text) == 16384


def test_use_case_profiles_use_deterministic_fixtures() -> None:
    scenarios = build_scenarios_for_benchmark(BenchmarkType.USE_CASE_PROFILES, 4096)
    scenarios_by_id = {item.scenario_id: item for item in scenarios}
    summary_payload = scenarios_by_id["use-case-profiles-long-document-summary-v1"].prompt_payload
    code_payload = scenarios_by_id["use-case-profiles-code-explanation-v1"].prompt_payload
    multi_turn_payload = scenarios_by_id["use-case-profiles-multi-turn-chat-v1"].prompt_payload

    assert set(scenarios_by_id) == {
        "use-case-profiles-quick-qa-v1",
        "use-case-profiles-long-document-summary-v1",
        "use-case-profiles-code-explanation-v1",
        "use-case-profiles-structured-extraction-v1",
        "use-case-profiles-long-form-generation-v1",
        "use-case-profiles-multi-turn-chat-v1",
    }
    assert summary_payload == TextPromptPayload(STATIC_SUMMARY_TEXT)
    assert code_payload == TextPromptPayload(STATIC_CODE_SAMPLE)
    assert isinstance(multi_turn_payload, MultiTurnChatPromptPayload)
    assert multi_turn_payload.turns == tuple(MULTI_TURN_CHAT_TURNS)
    assert multi_turn_payload.turns == (
        "Summarize the release note in one sentence.",
        "Rewrite it for a technical audience.",
        "Extract three action items.",
    )


def test_benchmark_family_resolution_keeps_ttft_as_distinct_execution_mode() -> None:
    smoke_family = resolve_benchmark_family(BenchmarkType.SMOKE)
    ttft_family = resolve_benchmark_family(BenchmarkType.TTFT)
    concurrency_family = resolve_benchmark_family(BenchmarkType.CONCURRENCY_SMOKE)

    smoke_scenarios = build_benchmark_scenarios(BenchmarkType.SMOKE, 4096)
    ttft_scenarios = build_benchmark_scenarios(BenchmarkType.TTFT, 4096)
    concurrency_scenarios = build_benchmark_scenarios(BenchmarkType.CONCURRENCY_SMOKE, 4096)

    assert smoke_family.execution_mode is ExecutionMode.GENERATE
    assert ttft_family.execution_mode is ExecutionMode.TTFT
    assert concurrency_family.execution_mode is ExecutionMode.CONCURRENCY
    assert smoke_scenarios[0].scenario_id == "smoke-basic-v1"
    assert ttft_scenarios[0].scenario_id == "ttft-basic-v1"
    assert [scenario.parallelism for scenario in concurrency_scenarios] == [2, 4]


def test_ttft_family_builds_three_deterministic_scenarios() -> None:
    scenarios = build_benchmark_scenarios(BenchmarkType.TTFT, 4096)

    assert [scenario.scenario_id for scenario in scenarios] == [
        "ttft-basic-v1",
        "ttft-chat-v1",
        "ttft-stream-shape-v1",
    ]
    stream_shape = scenarios[2]
    assert stream_shape.target_output_tokens == 32
    assert isinstance(stream_shape.prompt_payload, TextPromptPayload)
    assert "four short labeled lines" in stream_shape.prompt_payload.text


def test_stress_family_builds_two_deterministic_scenarios() -> None:
    scenarios = build_benchmark_scenarios(BenchmarkType.STRESS, 4096)

    assert [scenario.scenario_id for scenario in scenarios] == [
        "stress-sustained-load-v1",
        "stress-burst-load-v1",
    ]


def test_output_scaling_long_scenario_uses_release_target_band() -> None:
    scenarios = build_benchmark_scenarios(BenchmarkType.OUTPUT_SCALING, 4096)
    long_scenario = next(scenario for scenario in scenarios if scenario.scenario_id == "output-scaling-long-v1")

    assert long_scenario.target_output_tokens == 768


def test_concurrency_smoke_family_defines_small_parallelism_values() -> None:
    scenarios = build_benchmark_scenarios(BenchmarkType.CONCURRENCY_SMOKE, 4096)

    assert [scenario.scenario_id for scenario in scenarios] == [
        "concurrency-smoke-p2-v1",
        "concurrency-smoke-p4-v1",
    ]
    assert [scenario.parallelism for scenario in scenarios] == [2, 4]
    assert all(scenario.target_output_tokens == 32 for scenario in scenarios)


def test_context_scaling_scenarios_declare_calibration_metadata() -> None:
    scenarios = build_scenarios_for_benchmark(BenchmarkType.CONTEXT_SCALING, 4096)

    assert [scenario.prompt_template_version for scenario in scenarios] == ["v1", "v1", "v1"]
    assert [scenario.target_prompt_tokens for scenario in scenarios] == [1024, 2048, 3276]
    assert all(
        scenario.prompt_payload == TextPromptPayload(PROMPT_SCALING_BASE_TEXT)
        for scenario in scenarios
    )


def test_cold_warm_family_declares_requested_prep_behavior() -> None:
    scenarios = build_benchmark_scenarios(BenchmarkType.COLD_WARM, 4096)

    assert [(scenario.scenario_id, scenario.profile_tag) for scenario in scenarios] == [
        ("cold-warm-cold-start-v1", "cold_start"),
        ("cold-warm-warm-start-v1", "warm_start"),
    ]


def test_ollama_dispatcher_uses_stream_generate_for_ttft_and_measures_first_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned_run = PlannedRun(
        run_id="run-ttft-dispatch",
        run_index=1,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.TTFT,
        benchmark_type_index=1,
        scenario_id="ttft-basic-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="TTFT probe",
        scenario_version="v1",
    )
    scenario = build_benchmark_scenarios(BenchmarkType.TTFT, 4096)[0]

    class FakeClient:
        def stream_generate(self, *, model: str, prompt: str, options: dict[str, object]) -> list[dict[str, object]]:
            assert model == planned_run.model_name
            assert options["num_ctx"] == planned_run.context_size
            assert options["num_predict"] == 8
            return [
                {"response": "o", "done": False},
                {"response": "ok", "done": True, "eval_count": 4, "eval_duration": 250_000},
            ]

    perf_values = iter([10.0, 10.05, 10.2, 10.25])
    monkeypatch.setattr("ollama_workload_profiler.session.perf_counter", lambda: next(perf_values))

    result = _OllamaDispatcher(FakeClient()).execute(
        ExecutionRequest(
            run=planned_run,
            scenario=scenario,
            execution_mode=ExecutionMode.TTFT,
            execution_settings={"seed": 1234, "temperature": 0.2},
        )
    )

    assert result.metrics["ttft_ms"] == 50.0
    assert result.metrics["ttft_first_token_received"] is True
    assert result.metrics["stream_emission_count"] == 2
    assert result.metrics["stream_emission_offsets_ms"] == [50.0, 200.0]
    assert result.metrics["stream_duration_ms"] == 150.0
    assert result.metrics["stream_emission_interval_ms_median"] == 150.0
    assert result.metrics["stream_emission_interval_ms_p95"] == 150.0
    assert result.metrics["stream_output_units_per_second"] == 13.333
    assert result.metrics["stream_output_unit"] == "emission"
    assert result.metrics["tokens_per_second"] == 16000.0


def test_ollama_dispatcher_uses_emission_chunks_for_stream_shape_when_token_timing_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned_run = PlannedRun(
        run_id="run-ttft-stream-shape",
        run_index=1,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.TTFT,
        benchmark_type_index=1,
        scenario_id="ttft-basic-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="TTFT probe",
        scenario_version="v1",
    )
    scenario = build_benchmark_scenarios(BenchmarkType.TTFT, 4096)[0]

    class FakeClient:
        def stream_generate(self, *, model: str, prompt: str, options: dict[str, object]) -> list[dict[str, object]]:
            return [
                {"response": "", "done": False},
                {"response": "a", "done": False},
                {"thinking": "b", "done": False},
                {"response": "", "done": False},
                {"response": "c", "done": True, "eval_count": 3, "eval_duration": 300_000},
            ]

    perf_values = iter([100.0, 100.01, 100.04, 100.11, 100.2])
    monkeypatch.setattr("ollama_workload_profiler.session.perf_counter", lambda: next(perf_values))

    result = _OllamaDispatcher(FakeClient()).execute(
        ExecutionRequest(
            run=planned_run,
            scenario=scenario,
            execution_mode=ExecutionMode.TTFT,
            execution_settings={"seed": 1234, "temperature": 0.2},
        )
    )

    assert result.metrics["ttft_ms"] == 10.0
    assert result.metrics["stream_emission_count"] == 3
    assert result.metrics["stream_emission_offsets_ms"] == [10.0, 40.0, 110.0]
    assert result.metrics["stream_duration_ms"] == 100.0
    assert result.metrics["stream_emission_interval_ms_median"] == 50.0
    assert result.metrics["stream_emission_interval_ms_p95"] == 70.0
    assert result.metrics["stream_output_units_per_second"] == 30.0
    assert result.metrics["stream_output_unit"] == "emission"


def test_ollama_dispatcher_sets_interval_metrics_to_none_for_single_stream_emission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned_run = PlannedRun(
        run_id="run-ttft-single-emission",
        run_index=1,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.TTFT,
        benchmark_type_index=1,
        scenario_id="ttft-basic-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="TTFT probe",
        scenario_version="v1",
    )
    scenario = build_benchmark_scenarios(BenchmarkType.TTFT, 4096)[0]

    class FakeClient:
        def stream_generate(self, *, model: str, prompt: str, options: dict[str, object]) -> list[dict[str, object]]:
            return [{"response": "only", "done": True}]

    perf_values = iter([200.0, 200.025, 200.05])
    monkeypatch.setattr("ollama_workload_profiler.session.perf_counter", lambda: next(perf_values))

    result = _OllamaDispatcher(FakeClient()).execute(
        ExecutionRequest(
            run=planned_run,
            scenario=scenario,
            execution_mode=ExecutionMode.TTFT,
            execution_settings={"seed": 1234, "temperature": 0.2},
        )
    )

    assert result.metrics["stream_emission_count"] == 1
    assert result.metrics["stream_emission_offsets_ms"] == [25.0]
    assert result.metrics["stream_duration_ms"] == 0.0
    assert result.metrics["stream_emission_interval_ms_median"] is None
    assert result.metrics["stream_emission_interval_ms_p95"] is None
    assert result.metrics["stream_output_units_per_second"] is None


def test_concurrency_aggregate_metrics_report_request_elapsed_percentiles() -> None:
    metrics = _build_concurrency_aggregate_metrics(
        [
            {"request_index": 1, "elapsed_ms": 100.0, "ttft_ms": 10.0},
            {"request_index": 2, "elapsed_ms": 200.0, "ttft_ms": 20.0},
            {"request_index": 3, "elapsed_ms": 300.0, "ttft_ms": None},
            {"request_index": 4, "elapsed_ms": 400.0, "ttft_ms": 40.0},
        ]
    )

    assert metrics["concurrency_request_elapsed_ms_p50"] == 250.0
    assert metrics["concurrency_request_elapsed_ms_p95"] == 400.0
    assert metrics["concurrency_request_ttft_ms_p50"] == 20.0
    assert metrics["concurrency_request_ttft_ms_p95"] == 40.0


def test_ollama_dispatcher_runs_concurrency_smoke_as_parallel_streams() -> None:
    planned_run = PlannedRun(
        run_id="run-concurrency-smoke",
        run_index=1,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.CONCURRENCY_SMOKE,
        benchmark_type_index=1,
        scenario_id="concurrency-smoke-p2-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="Concurrency smoke p=2",
        scenario_version="v1",
    )
    scenario = build_benchmark_scenarios(BenchmarkType.CONCURRENCY_SMOKE, 4096)[0]

    class FakeWorkerClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def stream_generate(self, *, model: str, prompt: str, options: dict[str, object]) -> list[dict[str, object]]:
            self.calls.append({"model": model, "prompt": prompt, "options": options})
            return [
                {"response": "ok", "done": False},
                {"response": "done", "done": True, "eval_count": 2, "eval_duration": 250_000},
            ]

    worker_clients: list[FakeWorkerClient] = []

    def worker_client_factory() -> FakeWorkerClient:
        client = FakeWorkerClient()
        worker_clients.append(client)
        return client

    result = _OllamaDispatcher(
        object(),
        concurrency_client_factory=worker_client_factory,
    ).execute(
        ExecutionRequest(
            run=planned_run,
            scenario=scenario,
            execution_mode=ExecutionMode.CONCURRENCY,
            execution_settings={"seed": 1234, "temperature": 0.2},
        )
    )

    assert sum(len(client.calls) for client in worker_clients) == 2
    assert result.metrics["concurrency_parallelism"] == 2
    assert result.metrics["concurrency_mode"] == "same_machine_threaded_streams"
    assert result.metrics["concurrency_request_count"] == 2
    assert len(result.metrics["concurrency_requests"]) == 2
    assert all(
        request["ttft_ms"] is not None
        and request["elapsed_ms"] is not None
        and request["stream_emission_count"] == 2
        for request in result.metrics["concurrency_requests"]
    )
    assert result.metrics["concurrency_request_elapsed_ms_p50"] is not None
    assert result.metrics["concurrency_request_elapsed_ms_p95"] is not None
    assert result.metrics["concurrency_request_ttft_ms_p50"] is not None
    assert result.metrics["concurrency_request_ttft_ms_p95"] is not None


def test_ollama_dispatcher_uses_fresh_worker_clients_for_concurrency_smoke() -> None:
    planned_run = PlannedRun(
        run_id="run-concurrency-worker-clients",
        run_index=1,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.CONCURRENCY_SMOKE,
        benchmark_type_index=1,
        scenario_id="concurrency-smoke-p2-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="Concurrency smoke p=2",
        scenario_version="v1",
    )
    scenario = build_benchmark_scenarios(BenchmarkType.CONCURRENCY_SMOKE, 4096)[0]

    class SharedClient:
        def stream_generate(self, **_: object) -> list[dict[str, object]]:
            raise AssertionError("concurrency path must not use the shared dispatcher client")

    class WorkerClient:
        def __init__(self, worker_id: int) -> None:
            self.worker_id = worker_id
            self.closed = False

        def stream_generate(self, *, model: str, prompt: str, options: dict[str, object]) -> list[dict[str, object]]:
            return [
                {"response": f"worker-{self.worker_id}", "done": False},
                {"response": "done", "done": True},
            ]

        def close(self) -> None:
            self.closed = True

    created_clients: list[WorkerClient] = []

    def worker_client_factory() -> WorkerClient:
        client = WorkerClient(len(created_clients) + 1)
        created_clients.append(client)
        return client

    result = _OllamaDispatcher(
        SharedClient(),
        concurrency_client_factory=worker_client_factory,
    ).execute(
        ExecutionRequest(
            run=planned_run,
            scenario=scenario,
            execution_mode=ExecutionMode.CONCURRENCY,
            execution_settings={"seed": 1234, "temperature": 0.2},
        )
    )

    assert len(created_clients) == 2
    assert all(client.closed for client in created_clients)
    assert result.metrics["concurrency_request_count"] == 2


def test_response_metrics_extracts_load_and_prompt_generation_throughput() -> None:
    response = {
        "load_duration": 2_500_000,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 2_000_000,
        "eval_count": 4,
        "eval_duration": 500_000,
    }

    metrics = _response_metrics(response)

    assert metrics["load_duration"] == 2_500_000
    assert metrics["load_duration_ms"] == 2.5
    assert metrics["prompt_eval_count"] == 10
    assert metrics["prompt_eval_duration"] == 2_000_000
    assert metrics["prompt_tokens_per_second"] == 5000.0
    assert metrics["eval_count"] == 4
    assert metrics["eval_duration"] == 500_000
    assert metrics["generation_tokens_per_second"] == 8000.0
    assert metrics["tokens_per_second"] == 8000.0


def test_benchmark_runner_threads_execution_settings_into_dispatcher_request() -> None:
    planned_run = PlannedRun(
        run_id="run-request-settings",
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
    )
    execution_settings = {"seed": 1234, "temperature": 0.2, "top_p": 0.9}
    seen_requests: list[ExecutionRequest] = []

    class FakeSampler:
        def start(self, run: PlannedRun) -> None:
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[SamplePoint]:
            return []

    class FakeDispatcher:
        def execute(self, request: ExecutionRequest) -> ExecutionResult:
            seen_requests.append(request)
            return ExecutionResult(elapsed_ms=12.0, metrics={})

    runner = BenchmarkRunner(
        dispatcher=FakeDispatcher(),
        sampler_factory=FakeSampler,
        execution_settings=execution_settings,
    )

    runner.run(planned_run)

    assert len(seen_requests) == 1
    assert seen_requests[0].execution_settings == {
        "seed": 1234,
        "temperature": 0.2,
        "repetitions": 1,
        "top_p": 0.9,
    }


def test_benchmark_runner_defaults_missing_execution_settings_for_dispatcher() -> None:
    planned_run = PlannedRun(
        run_id="run-default-settings",
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
    )
    seen_requests: list[ExecutionRequest] = []

    class FakeSampler:
        def start(self, run: PlannedRun) -> None:
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[SamplePoint]:
            return []

    class FakeDispatcher:
        def execute(self, request: ExecutionRequest) -> ExecutionResult:
            seen_requests.append(request)
            return ExecutionResult(elapsed_ms=12.0, metrics={})

    runner = BenchmarkRunner(
        dispatcher=FakeDispatcher(),
        sampler_factory=FakeSampler,
    )

    runner.run(planned_run)

    assert seen_requests[0].execution_settings == {
        "seed": 42,
        "temperature": 0.0,
        "repetitions": 1,
    }


def test_direct_execution_request_defaults_missing_execution_settings() -> None:
    planned_run = PlannedRun(
        run_id="run-direct-request-defaults",
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
    )
    scenario = build_benchmark_scenarios(BenchmarkType.SMOKE, 4096)[0]

    request = ExecutionRequest(
        run=planned_run,
        scenario=scenario,
        execution_mode=ExecutionMode.GENERATE,
    )

    assert request.execution_settings == {
        "seed": 42,
        "temperature": 0.0,
        "repetitions": 1,
    }


def test_direct_execution_request_merges_sparse_execution_settings() -> None:
    planned_run = PlannedRun(
        run_id="run-direct-request-sparse",
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
    )
    scenario = build_benchmark_scenarios(BenchmarkType.SMOKE, 4096)[0]

    request = ExecutionRequest(
        run=planned_run,
        scenario=scenario,
        execution_mode=ExecutionMode.GENERATE,
        execution_settings={"top_p": 0.9},
    )

    assert request.execution_settings == {
        "seed": 42,
        "temperature": 0.0,
        "repetitions": 1,
        "top_p": 0.9,
    }


def test_benchmark_runner_merges_sparse_execution_settings_over_defaults() -> None:
    planned_run = PlannedRun(
        run_id="run-sparse-settings",
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
    )
    seen_requests: list[ExecutionRequest] = []

    class FakeSampler:
        def start(self, run: PlannedRun) -> None:
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[SamplePoint]:
            return []

    class FakeDispatcher:
        def execute(self, request: ExecutionRequest) -> ExecutionResult:
            seen_requests.append(request)
            return ExecutionResult(elapsed_ms=12.0, metrics={})

    runner = BenchmarkRunner(
        dispatcher=FakeDispatcher(),
        sampler_factory=FakeSampler,
        execution_settings={"top_p": 0.9},
    )

    runner.run(planned_run)

    assert seen_requests[0].execution_settings == {
        "seed": 42,
        "temperature": 0.0,
        "repetitions": 1,
        "top_p": 0.9,
    }


def test_benchmark_runner_rejects_malformed_execution_settings_before_dispatch() -> None:
    planned_run = PlannedRun(
        run_id="run-bad-runner-settings",
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
    )

    class FakeSampler:
        def start(self, run: PlannedRun) -> None:
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[SamplePoint]:
            return []

    class FakeDispatcher:
        def execute(self, request: ExecutionRequest) -> ExecutionResult:
            raise AssertionError("dispatcher should not be called")

    with pytest.raises(ValueError, match="execution_settings.top_p"):
        BenchmarkRunner(
            dispatcher=FakeDispatcher(),
            sampler_factory=FakeSampler,
            execution_settings={"top_p": "bad"},
        )


def test_ollama_dispatcher_builds_generation_options_from_execution_settings() -> None:
    planned_run = PlannedRun(
        run_id="run-generate-settings",
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
    )
    scenario = build_benchmark_scenarios(BenchmarkType.SMOKE, 4096)[0]

    request = ExecutionRequest(
        run=planned_run,
        scenario=scenario,
        execution_mode=ExecutionMode.GENERATE,
        execution_settings={"seed": 1234, "temperature": 0.2, "top_p": 0.9},
    )

    options = _OllamaDispatcher(client=object())._build_options(request)

    assert options == {
        "num_ctx": 4096,
        "num_predict": scenario.target_output_tokens,
        "seed": 1234,
        "temperature": 0.2,
        "top_p": 0.9,
    }


def test_build_profile_session_plan_defaults_required_execution_settings() -> None:
    plan = build_profile_session_plan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE],
    )

    assert plan.execution_settings["seed"] == 42
    assert plan.execution_settings["temperature"] == 0.0
    assert plan.execution_settings["repetitions"] == 1
    assert "top_p" not in plan.execution_settings or plan.execution_settings["top_p"] is None


def test_run_profile_session_uses_default_execution_settings_from_profile_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = build_profile_session_plan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE],
    )
    observed_requests: list[ExecutionRequest] = []

    class FakeRunner:
        def __init__(self, *, execution_settings: dict[str, object], **_: object) -> None:
            assert execution_settings == {
                "seed": 42,
                "temperature": 0.0,
                "repetitions": 1,
            }

        def run(self, planned_run: PlannedRun) -> RunResult:
            scenario = build_benchmark_scenarios(BenchmarkType.SMOKE, planned_run.context_size)[0]
            observed_requests.append(
                ExecutionRequest(
                    run=planned_run,
                    scenario=scenario,
                    execution_mode=ExecutionMode.GENERATE,
                    execution_settings=plan.execution_settings,
                )
            )
            return RunResult(
                run_id=planned_run.run_id,
                run_index=planned_run.run_index,
                model_name=planned_run.model_name,
                context_size=planned_run.context_size,
                context_index=planned_run.context_index,
                benchmark_type=planned_run.benchmark_type,
                benchmark_type_index=planned_run.benchmark_type_index,
                scenario_id=planned_run.scenario_id,
                scenario_index=planned_run.scenario_index,
                scenario_version=planned_run.scenario_version,
                repetition_index=planned_run.repetition_index,
                scenario_name=planned_run.scenario_name,
                state=RunState.COMPLETED,
                elapsed_ms=10.0,
                metrics={"tokens_per_second": 5.0},
            )

    class FakeClient:
        def list_models(self) -> list[str]:
            return [plan.model_name]

    monkeypatch.setattr("ollama_workload_profiler.session.BenchmarkRunner", FakeRunner)

    result = run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        session_timestamp=datetime(2026, 4, 19, 10, 0, 0, tzinfo=timezone.utc),
    )

    assert observed_requests[0].execution_settings == {
        "seed": 42,
        "temperature": 0.0,
        "repetitions": 1,
    }
    assert result.summary.session_metrics["completed_runs"] == 1


def test_direct_benchmark_session_plan_normalizes_sparse_execution_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE],
        execution_settings={"repetitions": 2},
    )
    observed_requests: list[ExecutionRequest] = []

    class FakeRunner:
        def __init__(self, *, execution_settings: dict[str, object], **_: object) -> None:
            assert execution_settings == {
                "seed": 42,
                "temperature": 0.0,
                "repetitions": 2,
            }

        def run(self, planned_run: PlannedRun) -> RunResult:
            scenario = build_benchmark_scenarios(BenchmarkType.SMOKE, planned_run.context_size)[0]
            observed_requests.append(
                ExecutionRequest(
                    run=planned_run,
                    scenario=scenario,
                    execution_mode=ExecutionMode.GENERATE,
                    execution_settings=plan.execution_settings,
                )
            )
            return RunResult(
                run_id=planned_run.run_id,
                run_index=planned_run.run_index,
                model_name=planned_run.model_name,
                context_size=planned_run.context_size,
                context_index=planned_run.context_index,
                benchmark_type=planned_run.benchmark_type,
                benchmark_type_index=planned_run.benchmark_type_index,
                scenario_id=planned_run.scenario_id,
                scenario_index=planned_run.scenario_index,
                scenario_version=planned_run.scenario_version,
                repetition_index=planned_run.repetition_index,
                scenario_name=planned_run.scenario_name,
                state=RunState.COMPLETED,
                elapsed_ms=10.0,
                metrics={"tokens_per_second": 5.0},
            )

    class FakeClient:
        def list_models(self) -> list[str]:
            return [plan.model_name]

    monkeypatch.setattr("ollama_workload_profiler.session.BenchmarkRunner", FakeRunner)

    result = run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        session_timestamp=datetime(2026, 4, 19, 10, 5, 0, tzinfo=timezone.utc),
    )

    assert plan.execution_settings == {
        "seed": 42,
        "temperature": 0.0,
        "repetitions": 2,
    }
    assert observed_requests[0].execution_settings == plan.execution_settings
    assert result.summary.session_metrics["completed_runs"] == 2


def test_direct_benchmark_session_plan_applies_defaults_when_execution_settings_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE],
    )
    observed_requests: list[ExecutionRequest] = []

    class FakeRunner:
        def __init__(self, *, execution_settings: dict[str, object], **_: object) -> None:
            assert execution_settings == {
                "seed": 42,
                "temperature": 0.0,
                "repetitions": 1,
            }

        def run(self, planned_run: PlannedRun) -> RunResult:
            scenario = build_benchmark_scenarios(BenchmarkType.SMOKE, planned_run.context_size)[0]
            observed_requests.append(
                ExecutionRequest(
                    run=planned_run,
                    scenario=scenario,
                    execution_mode=ExecutionMode.GENERATE,
                    execution_settings=plan.execution_settings,
                )
            )
            return RunResult(
                run_id=planned_run.run_id,
                run_index=planned_run.run_index,
                model_name=planned_run.model_name,
                context_size=planned_run.context_size,
                context_index=planned_run.context_index,
                benchmark_type=planned_run.benchmark_type,
                benchmark_type_index=planned_run.benchmark_type_index,
                scenario_id=planned_run.scenario_id,
                scenario_index=planned_run.scenario_index,
                scenario_version=planned_run.scenario_version,
                repetition_index=planned_run.repetition_index,
                scenario_name=planned_run.scenario_name,
                state=RunState.COMPLETED,
                elapsed_ms=10.0,
                metrics={"tokens_per_second": 5.0},
            )

    class FakeClient:
        def list_models(self) -> list[str]:
            return [plan.model_name]

    monkeypatch.setattr("ollama_workload_profiler.session.BenchmarkRunner", FakeRunner)

    result = run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        session_timestamp=datetime(2026, 4, 19, 10, 10, 0, tzinfo=timezone.utc),
    )

    assert plan.execution_settings == {
        "seed": 42,
        "temperature": 0.0,
        "repetitions": 1,
    }
    assert observed_requests[0].execution_settings == plan.execution_settings
    assert result.summary.session_metrics["completed_runs"] == 1


@pytest.mark.parametrize(
    ("execution_settings", "expected_message"),
    [
        ({"seed": "bad"}, "execution_settings.seed"),
        ({"temperature": "bad"}, "execution_settings.temperature"),
        ({"top_p": "bad"}, "execution_settings.top_p"),
        ({"seed": True}, "execution_settings.seed"),
        ({"temperature": -0.1}, "execution_settings.temperature"),
        ({"temperature": 2.1}, "execution_settings.temperature"),
        ({"top_p": -0.1}, "execution_settings.top_p"),
        ({"top_p": 1.1}, "execution_settings.top_p"),
    ],
)
def test_benchmark_session_plan_rejects_malformed_execution_settings_values(
    execution_settings: dict[str, object],
    expected_message: str,
) -> None:
    with pytest.raises(ValidationError, match=expected_message):
        BenchmarkSessionPlan(
            model_name="llama3.2",
            contexts=[4096],
            benchmark_types=[BenchmarkType.SMOKE],
            execution_settings=execution_settings,
        )


def test_benchmark_session_plan_rejects_unknown_execution_settings_keys() -> None:
    with pytest.raises(ValidationError, match="execution_settings.tempertaure"):
        BenchmarkSessionPlan(
            model_name="llama3.2",
            contexts=[4096],
            benchmark_types=[BenchmarkType.SMOKE],
            execution_settings={"tempertaure": 0.2},
        )


@pytest.mark.parametrize(
    ("execution_settings", "expected_message"),
    [
        ({"warmup_runs": 0}, "execution_settings.warmup_runs"),
        ({"warmup_runs": -1}, "execution_settings.warmup_runs"),
        ({"warmup_runs": "bad"}, "execution_settings.warmup_runs"),
        ({"warmup_enabled": "bad"}, "execution_settings.warmup_enabled"),
        ({"warmup_enabled": 1}, "execution_settings.warmup_enabled"),
    ],
)
def test_benchmark_session_plan_rejects_invalid_warmup_execution_settings(
    execution_settings: dict[str, object],
    expected_message: str,
) -> None:
    with pytest.raises(ValidationError, match=expected_message):
        BenchmarkSessionPlan(
            model_name="llama3.2",
            contexts=[4096],
            benchmark_types=[BenchmarkType.SMOKE],
            execution_settings=execution_settings,
        )


def test_ollama_dispatcher_omits_top_p_when_not_requested() -> None:
    planned_run = PlannedRun(
        run_id="run-ttft-settings",
        run_index=1,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.TTFT,
        benchmark_type_index=1,
        scenario_id="ttft-basic-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="TTFT probe",
    )
    scenario = build_benchmark_scenarios(BenchmarkType.TTFT, 4096)[0]

    request = ExecutionRequest(
        run=planned_run,
        scenario=scenario,
        execution_mode=ExecutionMode.TTFT,
        execution_settings={"seed": 4321, "temperature": 0.0},
    )

    options = _OllamaDispatcher(client=object())._build_options(request)

    assert options["num_ctx"] == 4096
    assert options["num_predict"] == 8
    assert options["seed"] == 4321
    assert options["temperature"] == 0.0
    assert "top_p" not in options


def test_ollama_dispatcher_records_missing_first_token_for_empty_ttft_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned_run = PlannedRun(
        run_id="run-ttft-empty",
        run_index=2,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.TTFT,
        benchmark_type_index=1,
        scenario_id="ttft-basic-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="TTFT probe",
        scenario_version="v1",
    )
    scenario = build_benchmark_scenarios(BenchmarkType.TTFT, 4096)[0]

    class FakeClient:
        def stream_generate(self, *, model: str, prompt: str, options: dict[str, object]) -> list[dict[str, object]]:
            return []

    perf_values = iter([20.0, 20.1])
    monkeypatch.setattr("ollama_workload_profiler.session.perf_counter", lambda: next(perf_values))

    result = _OllamaDispatcher(FakeClient()).execute(
        ExecutionRequest(
            run=planned_run,
            scenario=scenario,
            execution_mode=ExecutionMode.TTFT,
            execution_settings={"seed": 1234, "temperature": 0.2},
        )
    )

    assert "ttft_ms" not in result.metrics
    assert result.metrics["ttft_first_token_received"] is False
    assert result.metrics["stream_emission_count"] == 0
    assert result.metrics["stream_emission_offsets_ms"] == []
    assert result.metrics["stream_duration_ms"] is None
    assert result.metrics["stream_emission_interval_ms_median"] is None
    assert result.metrics["stream_emission_interval_ms_p95"] is None
    assert result.metrics["stream_output_units_per_second"] is None
    assert result.elapsed_ms == 100.0


def test_ollama_dispatcher_uses_stream_chat_for_chat_ttft_scenarios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned_run = PlannedRun(
        run_id="run-ttft-chat",
        run_index=3,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.TTFT,
        benchmark_type_index=1,
        scenario_id="ttft-chat-v1",
        scenario_index=2,
        repetition_index=1,
        scenario_name="TTFT chat probe",
        scenario_version="v1",
    )
    scenario = build_benchmark_scenarios(BenchmarkType.TTFT, 4096)[1]
    stream_chat_calls: list[list[dict[str, object]]] = []

    class FakeClient:
        def stream_chat(self, *, model: str, messages: list[dict[str, object]], options: dict[str, object]) -> list[dict[str, object]]:
            assert model == planned_run.model_name
            assert options["num_ctx"] == planned_run.context_size
            assert options["num_predict"] == 8
            stream_chat_calls.append(messages)
            return [
                {"message": {"content": "", "thinking": "Thinking"}, "done": False},
                {"message": {"content": "", "thinking": ""}, "done": True},
            ]

    perf_values = iter([30.0, 30.02, 30.1])
    monkeypatch.setattr("ollama_workload_profiler.session.perf_counter", lambda: next(perf_values))

    result = _OllamaDispatcher(FakeClient()).execute(
        ExecutionRequest(
            run=planned_run,
            scenario=scenario,
            execution_mode=ExecutionMode.TTFT,
            execution_settings={"seed": 1234, "temperature": 0.2},
        )
    )

    assert len(stream_chat_calls) == 1
    assert [message["content"] for message in stream_chat_calls[0]] == [
        "Reply with a single token.",
        "Reply again with the same single token.",
    ]
    assert result.metrics["ttft_ms"] == 20.0
    assert result.metrics["ttft_first_token_received"] is True


def test_ollama_dispatcher_does_not_count_empty_non_content_ttft_chunks_as_first_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned_run = PlannedRun(
        run_id="run-ttft-empty-chat",
        run_index=4,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.TTFT,
        benchmark_type_index=1,
        scenario_id="ttft-chat-v1",
        scenario_index=2,
        repetition_index=1,
        scenario_name="TTFT chat probe",
        scenario_version="v1",
    )
    scenario = build_benchmark_scenarios(BenchmarkType.TTFT, 4096)[1]

    class FakeClient:
        def stream_chat(self, *, model: str, messages: list[dict[str, object]], options: dict[str, object]) -> list[dict[str, object]]:
            return [
                {"message": {"content": "", "thinking": ""}, "done": False},
                {"message": {"content": "", "thinking": ""}, "done": True},
            ]

    perf_values = iter([40.0, 40.1])
    monkeypatch.setattr("ollama_workload_profiler.session.perf_counter", lambda: next(perf_values))

    result = _OllamaDispatcher(FakeClient()).execute(
        ExecutionRequest(
            run=planned_run,
            scenario=scenario,
            execution_mode=ExecutionMode.TTFT,
            execution_settings={"seed": 1234, "temperature": 0.2},
        )
    )

    assert "ttft_ms" not in result.metrics
    assert result.metrics["ttft_first_token_received"] is False


def test_benchmark_runner_finalizes_completed_run_from_execution_facts() -> None:
    planned_run = PlannedRun(
        run_id="run-1234",
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
    )
    state_transitions: list[RunState] = []
    dispatched_modes: list[ExecutionMode] = []

    class FakeSampler:
        def __init__(self) -> None:
            self.started = False
            self.stopped = False

        def start(self, run: PlannedRun) -> None:
            self.started = True
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[SamplePoint]:
            self.stopped = True
            return [SamplePoint(phase="generation", rss_mb=128.0, cpu_percent=44.0)]

    class FakeDispatcher:
        def execute(self, request: object) -> ExecutionResult:
            dispatched_modes.append(request.execution_mode)
            return ExecutionResult(
                elapsed_ms=42.5,
                metrics={"tokens_per_second": 12.0},
            )

    sampler = FakeSampler()
    runner = BenchmarkRunner(
        dispatcher=FakeDispatcher(),
        sampler_factory=lambda: sampler,
        on_state_change=lambda _run, state: state_transitions.append(state),
    )

    result = runner.run(planned_run)

    assert sampler.started is True
    assert sampler.stopped is True
    assert dispatched_modes == [ExecutionMode.GENERATE]
    assert state_transitions == [RunState.STARTING, RunState.COMPLETED]
    assert result.state is RunState.COMPLETED
    assert result.elapsed_ms == 42.5
    assert result.metrics["tokens_per_second"] == 12.0
    assert result.metrics["phase_peaks"]["generation"]["rss_mb"] == 128.0


def test_benchmark_runner_adds_best_effort_gpu_summary_from_sampler_samples() -> None:
    planned_run = PlannedRun(
        run_id="run-gpu-summary",
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
    )

    class FakeSampler:
        def start(self, run: PlannedRun) -> None:
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[SamplePoint]:
            return [
                SamplePoint(
                    phase="load",
                    rss_mb=100.0,
                    cpu_percent=10.0,
                    gpu_telemetry_available=True,
                    gpu_telemetry_source="nvidia-smi",
                    gpu_memory_used_mb=1024.0,
                    gpu_util_percent=20.0,
                    gpu_device_count=1,
                    gpu_telemetry_notes=[],
                ),
                SamplePoint(
                    phase="generation",
                    rss_mb=120.0,
                    cpu_percent=30.0,
                    gpu_telemetry_available=True,
                    gpu_telemetry_source="nvidia-smi",
                    gpu_memory_used_mb=1536.0,
                    gpu_util_percent=80.0,
                    gpu_device_count=1,
                    gpu_telemetry_notes=[],
                ),
            ]

    class FakeDispatcher:
        def execute(self, request: object) -> ExecutionResult:
            return ExecutionResult(elapsed_ms=42.5, metrics={})

    runner = BenchmarkRunner(dispatcher=FakeDispatcher(), sampler_factory=FakeSampler)

    result = runner.run(planned_run)

    assert result.metrics["gpu_telemetry_available"] is True
    assert result.metrics["gpu_telemetry_source"] == "nvidia-smi"
    assert result.metrics["gpu_backend"] == "nvidia-smi"
    assert result.metrics["peak_gpu_memory_mb"] == 1536.0
    assert result.metrics["avg_gpu_util_percent"] == 50.0
    assert result.metrics["peak_gpu_util_percent"] == 80.0
    assert result.metrics["gpu_device_count"] == 1


def test_benchmark_runner_adds_sampled_process_summary_from_sampler_samples() -> None:
    planned_run = PlannedRun(
        run_id="run-process-summary",
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
    )

    class FakeSampler:
        def start(self, run: PlannedRun) -> None:
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[SamplePoint]:
            return [
                SamplePoint(
                    phase="load",
                    rss_mb=100.0,
                    cpu_percent=10.0,
                    sampled_process_count=1,
                    sampled_process_ids=[10],
                ),
                SamplePoint(
                    phase="generation",
                    rss_mb=120.0,
                    cpu_percent=30.0,
                    sampled_process_count=3,
                    sampled_process_ids=[10, 20, 30],
                ),
            ]

    class FakeDispatcher:
        def execute(self, request: object) -> ExecutionResult:
            return ExecutionResult(elapsed_ms=42.5, metrics={})

    runner = BenchmarkRunner(dispatcher=FakeDispatcher(), sampler_factory=FakeSampler)

    result = runner.run(planned_run)

    assert result.metrics["sampled_process_count"] == 3
    assert result.metrics["sampled_process_ids"] == [10, 20, 30]


def test_benchmark_runner_process_summary_count_matches_unique_sampled_process_ids() -> None:
    planned_run = PlannedRun(
        run_id="run-process-summary-dedup",
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
    )

    class FakeSampler:
        def start(self, run: PlannedRun) -> None:
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[SamplePoint]:
            return [
                SamplePoint(
                    phase="load",
                    rss_mb=100.0,
                    cpu_percent=10.0,
                    sampled_process_count=2,
                    sampled_process_ids=[10, 10],
                ),
                SamplePoint(
                    phase="generation",
                    rss_mb=120.0,
                    cpu_percent=30.0,
                    sampled_process_count=2,
                    sampled_process_ids=[10, 20],
                ),
            ]

    class FakeDispatcher:
        def execute(self, request: object) -> ExecutionResult:
            return ExecutionResult(elapsed_ms=42.5, metrics={})

    runner = BenchmarkRunner(dispatcher=FakeDispatcher(), sampler_factory=FakeSampler)

    result = runner.run(planned_run)

    assert result.metrics["sampled_process_ids"] == [10, 20]
    assert result.metrics["sampled_process_count"] == len(result.metrics["sampled_process_ids"])


def test_benchmark_runner_records_gpu_telemetry_unavailable_without_failing_run() -> None:
    planned_run = PlannedRun(
        run_id="run-gpu-unavailable",
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
    )

    class FakeSampler:
        def start(self, run: PlannedRun) -> None:
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[SamplePoint]:
            return [
                SamplePoint(
                    phase="generation",
                    rss_mb=120.0,
                    cpu_percent=30.0,
                    gpu_telemetry_available=False,
                    gpu_telemetry_source=None,
                    gpu_telemetry_notes=["nvidia-smi not found"],
                )
            ]

    class FakeDispatcher:
        def execute(self, request: object) -> ExecutionResult:
            return ExecutionResult(elapsed_ms=42.5, metrics={})

    runner = BenchmarkRunner(dispatcher=FakeDispatcher(), sampler_factory=FakeSampler)

    result = runner.run(planned_run)

    assert result.state is RunState.COMPLETED
    assert result.metrics["gpu_telemetry_available"] is False
    assert result.metrics["gpu_telemetry_source"] is None
    assert result.metrics["gpu_backend"] is None
    assert result.metrics["peak_gpu_memory_mb"] is None
    assert result.metrics["avg_gpu_util_percent"] is None
    assert result.metrics["peak_gpu_util_percent"] is None
    assert result.metrics["gpu_telemetry_notes"] == ["nvidia-smi not found"]


def test_benchmark_runner_uses_ttft_execution_mode_and_classifies_stopped_runs() -> None:
    planned_run = PlannedRun(
        run_id="run-ttft",
        run_index=2,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.TTFT,
        benchmark_type_index=1,
        scenario_id="ttft-basic-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="TTFT probe",
    )
    state_transitions: list[RunState] = []
    dispatched_modes: list[ExecutionMode] = []

    class FakeSampler:
        def start(self, run: PlannedRun) -> None:
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[SamplePoint]:
            return []

    class FakeDispatcher:
        def execute(self, request: object) -> ExecutionResult:
            dispatched_modes.append(request.execution_mode)
            raise BenchmarkExecutionStopped("cancelled")

    runner = BenchmarkRunner(
        dispatcher=FakeDispatcher(),
        sampler_factory=FakeSampler,
        on_state_change=lambda _run, state: state_transitions.append(state),
    )

    result = runner.run(planned_run)

    assert dispatched_modes == [ExecutionMode.TTFT]
    assert state_transitions == [RunState.STARTING, RunState.STOPPED]
    assert result.state is RunState.STOPPED
    assert result.metrics["stop_reason"] == "cancelled"


def test_benchmark_runner_returns_failed_result_when_sampler_stop_raises() -> None:
    planned_run = PlannedRun(
        run_id="run-stop-error",
        run_index=3,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.SMOKE,
        benchmark_type_index=1,
        scenario_id="smoke-basic-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="Smoke sanity check",
    )
    state_transitions: list[RunState] = []

    class FakeSampler:
        def start(self, run: PlannedRun) -> None:
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[SamplePoint]:
            raise RuntimeError("sampler shutdown failed")

    class FakeDispatcher:
        def execute(self, request: object) -> ExecutionResult:
            assert request.execution_mode is ExecutionMode.GENERATE
            return ExecutionResult(
                elapsed_ms=19.5,
                metrics={"tokens_per_second": 8.0},
            )

    runner = BenchmarkRunner(
        dispatcher=FakeDispatcher(),
        sampler_factory=FakeSampler,
        on_state_change=lambda _run, state: state_transitions.append(state),
    )

    result = runner.run(planned_run)

    assert state_transitions == [RunState.STARTING, RunState.FAILED]
    assert result.state is RunState.FAILED
    assert result.elapsed_ms == 19.5
    assert result.metrics["tokens_per_second"] == 8.0
    assert result.metrics["execution_outcome_state"] == RunState.COMPLETED.value
    assert result.metrics["finalization_error"] == "sampler shutdown failed"
    assert result.failure is not None
    assert result.failure.kind == "failed"
    assert result.failure.message == "sampler shutdown failed"


def test_benchmark_runner_returns_failed_result_when_telemetry_is_malformed() -> None:
    planned_run = PlannedRun(
        run_id="run-bad-telemetry",
        run_index=4,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.TTFT,
        benchmark_type_index=1,
        scenario_id="ttft-basic-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="TTFT probe",
    )
    state_transitions: list[RunState] = []

    class FakeSampler:
        def start(self, run: PlannedRun) -> None:
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[dict[str, object]]:
            return [{"phase": None, "rss_mb": 100.0, "cpu_percent": 5.0}]

    class FakeDispatcher:
        def execute(self, request: object) -> ExecutionResult:
            assert request.execution_mode is ExecutionMode.TTFT
            raise BenchmarkExecutionStopped("cancelled")

    runner = BenchmarkRunner(
        dispatcher=FakeDispatcher(),
        sampler_factory=FakeSampler,
        on_state_change=lambda _run, state: state_transitions.append(state),
    )

    result = runner.run(planned_run)

    assert state_transitions == [RunState.STARTING, RunState.FAILED]
    assert result.state is RunState.FAILED
    assert result.metrics["stop_reason"] == "cancelled"
    assert result.metrics["execution_outcome_state"] == RunState.STOPPED.value
    assert result.metrics["finalization_error"] == "sample.phase must be a non-empty string"
    assert result.failure is not None
    assert result.failure.kind == "failed"
    assert result.failure.message == "sample.phase must be a non-empty string"


def test_benchmark_runner_persists_cold_warm_prep_metadata() -> None:
    planned_run = PlannedRun(
        run_id="run-cold-warm",
        run_index=5,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.COLD_WARM,
        benchmark_type_index=1,
        scenario_id="cold-warm-cold-start-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="Cold start probe",
        scenario_version="v1",
    )

    class FakeSampler:
        def start(self, run: PlannedRun) -> None:
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[SamplePoint]:
            return []

    class FakeDispatcher:
        def execute(self, request: object) -> ExecutionResult:
            return ExecutionResult(elapsed_ms=33.0, metrics={})

    runner = BenchmarkRunner(
        dispatcher=FakeDispatcher(),
        sampler_factory=FakeSampler,
    )

    result = runner.run(planned_run)

    assert result.state is RunState.COMPLETED
    assert result.metrics["requested_prep_behavior"] == "cold_start"
    assert result.metrics["actual_prep_method"] == "scenario_declared"


def test_benchmark_runner_preserves_cold_warm_prep_metadata_for_stopped_runs() -> None:
    planned_run = PlannedRun(
        run_id="run-cold-warm-stopped",
        run_index=6,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.COLD_WARM,
        benchmark_type_index=1,
        scenario_id="cold-warm-cold-start-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="Cold start probe",
        scenario_version="v1",
    )

    class FakeSampler:
        def start(self, run: PlannedRun) -> None:
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[SamplePoint]:
            return []

    class FakeDispatcher:
        def execute(self, request: object) -> ExecutionResult:
            raise BenchmarkExecutionStopped("cancelled")

    runner = BenchmarkRunner(
        dispatcher=FakeDispatcher(),
        sampler_factory=FakeSampler,
    )

    result = runner.run(planned_run)

    assert result.state is RunState.STOPPED
    assert result.metrics["requested_prep_behavior"] == "cold_start"
    assert result.metrics["actual_prep_method"] == "scenario_declared"


def test_benchmark_runner_preserves_cold_warm_prep_metadata_when_sampler_stop_fails() -> None:
    planned_run = PlannedRun(
        run_id="run-cold-warm-finalization-failed",
        run_index=7,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.COLD_WARM,
        benchmark_type_index=1,
        scenario_id="cold-warm-cold-start-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="Cold start probe",
        scenario_version="v1",
    )

    class FakeSampler:
        def start(self, run: PlannedRun) -> None:
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[SamplePoint]:
            raise RuntimeError("sampler stop failed")

    class FakeDispatcher:
        def execute(self, request: object) -> ExecutionResult:
            return ExecutionResult(elapsed_ms=12.0, metrics={})

    runner = BenchmarkRunner(
        dispatcher=FakeDispatcher(),
        sampler_factory=FakeSampler,
    )

    result = runner.run(planned_run)

    assert result.state is RunState.FAILED
    assert result.failure is not None
    assert result.failure.phase == "finalization"
    assert result.metrics["requested_prep_behavior"] == "cold_start"
    assert result.metrics["actual_prep_method"] == "scenario_declared"
    assert result.metrics["finalization_error"] == "sampler stop failed"


def test_run_profile_session_enforces_cold_and_warm_prep_before_measured_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.COLD_WARM],
        execution_settings={
            "repetitions": 1,
            "warmup_runs": 1,
            "warmup_enabled": True,
        },
    )
    call_log: list[str] = []

    class FakeRunner:
        def __init__(self, **_: object) -> None:
            return None

        def run(self, planned_run: PlannedRun) -> RunResult:
            call_log.append(f"run:{planned_run.scenario_id}")
            return RunResult(
                run_id=planned_run.run_id,
                run_index=planned_run.run_index,
                model_name=planned_run.model_name,
                context_size=planned_run.context_size,
                context_index=planned_run.context_index,
                benchmark_type=planned_run.benchmark_type,
                benchmark_type_index=planned_run.benchmark_type_index,
                scenario_id=planned_run.scenario_id,
                scenario_index=planned_run.scenario_index,
                scenario_version=planned_run.scenario_version,
                repetition_index=planned_run.repetition_index,
                scenario_name=planned_run.scenario_name,
                state=RunState.COMPLETED,
                elapsed_ms=10.0,
                metrics={"tokens_per_second": 5.0},
            )

    class FakeClient:
        def list_models(self) -> list[str]:
            call_log.append("list_models")
            return [plan.model_name]

        def unload_model(self, *, model: str) -> dict[str, object]:
            call_log.append(f"unload:{model}")
            return {"done_reason": "unload"}

        def preload_model(self, *, model: str, options: dict[str, object] | None = None) -> dict[str, object]:
            assert options == {
                "num_ctx": 4096,
                "num_predict": 16,
                "seed": 42,
                "temperature": 0.0,
            }
            call_log.append(f"preload:{model}")
            return {"done": True}

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            call_log.append(f"warmup:{options['num_ctx']}")
            return {
                "response": "ok",
                "total_duration": 1_000_000,
                "prompt_eval_count": 1,
                "prompt_eval_duration": 1_000_000,
                "eval_count": 1,
                "eval_duration": 1_000_000,
            }

    monkeypatch.setattr("ollama_workload_profiler.session.BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(
        "ollama_workload_profiler.session._build_host_metadata",
        lambda: {
            "hostname": "bench-host",
            "os": {"platform": "test-os", "release": "1.0"},
            "cpu": {"logical_cores": 16, "physical_cores": 8},
            "memory": {"total_mb": 32768.0},
        },
    )
    monkeypatch.setattr(
        "ollama_workload_profiler.session._build_run_system_snapshot",
        lambda: {
            "cpu_percent": 21.5,
            "memory_available_mb": 24000.0,
            "memory_used_percent": 26.8,
            "ollama_process_count": 2,
        },
    )

    result = run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        session_timestamp=datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert call_log == [
        "list_models",
        "unload:llama3.2",
        "run:cold-warm-cold-start-v1",
        "preload:llama3.2",
        "run:cold-warm-warm-start-v1",
    ]
    assert result.runs[0].metrics["requested_prep_behavior"] == "cold_start"
    assert result.runs[0].metrics["actual_prep_method"] == "explicit_unload"
    assert result.runs[0].metrics["prep_enforcement_succeeded"] is True
    assert result.runs[1].metrics["requested_prep_behavior"] == "warm_start"
    assert result.runs[1].metrics["actual_prep_method"] == "explicit_preload"
    assert result.runs[1].metrics["prep_enforcement_succeeded"] is True


def test_context_scaling_runs_record_requested_and_actual_prompt_tokens(
    tmp_path: Path,
) -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.CONTEXT_SCALING],
        execution_settings={
            "repetitions": 1,
            "warmup_runs": 1,
            "warmup_enabled": False,
        },
    )
    scenario = build_benchmark_scenarios(BenchmarkType.CONTEXT_SCALING, 4096)[0]
    expanded_plan = [
        PlannedRun(
            run_id="run-context-calibration",
            run_index=1,
            model_name=plan.model_name,
            context_size=4096,
            context_index=1,
            benchmark_type=BenchmarkType.CONTEXT_SCALING,
            benchmark_type_index=1,
            scenario_id=scenario.scenario_id,
            scenario_index=1,
            repetition_index=1,
            scenario_name=scenario.name,
            scenario_version=scenario.version,
        )
    ]
    call_log: list[int] = []

    class FakeClient:
        def list_models(self) -> list[str]:
            return [plan.model_name]

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            call_log.append(int(options["num_predict"]))
            return {
                "response": "ok",
                "total_duration": 1_000_000,
                "prompt_eval_count": 1024,
                "prompt_eval_duration": 1_000_000_000,
                "eval_count": 4,
                "eval_duration": 500_000_000,
            }

    result = run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        expanded_plan=expanded_plan,
        session_timestamp=datetime(2026, 4, 19, 13, 15, 0, tzinfo=timezone.utc),
    )

    calibrated_run = result.runs[0]
    assert call_log == [1, 64]
    assert calibrated_run.metrics["requested_fill_ratio"] == 0.25
    assert calibrated_run.metrics["target_prompt_tokens"] == 1024
    assert calibrated_run.metrics["actual_prompt_tokens"] == 1024
    assert calibrated_run.metrics["calibration_status"] == "exact"
    assert calibrated_run.metrics["calibration_attempts"] == 1
    assert calibrated_run.metrics["calibration_cache_hit"] is False
    assert calibrated_run.metrics["eligible_for_strict_aggregate"] is True
    assert calibrated_run.metrics["eligible_for_calibrated_context_aggregate"] is True


def test_context_calibration_is_cached_for_repeated_runs(
    tmp_path: Path,
) -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.CONTEXT_SCALING],
        execution_settings={
            "repetitions": 2,
            "warmup_runs": 1,
            "warmup_enabled": False,
        },
    )
    scenario = build_benchmark_scenarios(BenchmarkType.CONTEXT_SCALING, 4096)[0]
    expanded_plan = [
        PlannedRun(
            run_id="run-context-calibration-1",
            run_index=1,
            model_name=plan.model_name,
            context_size=4096,
            context_index=1,
            benchmark_type=BenchmarkType.CONTEXT_SCALING,
            benchmark_type_index=1,
            scenario_id=scenario.scenario_id,
            scenario_index=1,
            repetition_index=1,
            scenario_name=scenario.name,
            scenario_version=scenario.version,
        ),
        PlannedRun(
            run_id="run-context-calibration-2",
            run_index=2,
            model_name=plan.model_name,
            context_size=4096,
            context_index=1,
            benchmark_type=BenchmarkType.CONTEXT_SCALING,
            benchmark_type_index=1,
            scenario_id=scenario.scenario_id,
            scenario_index=1,
            repetition_index=2,
            scenario_name=scenario.name,
            scenario_version=scenario.version,
        ),
    ]
    call_log: list[int] = []

    class FakeClient:
        def list_models(self) -> list[str]:
            return [plan.model_name]

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            call_log.append(int(options["num_predict"]))
            return {
                "response": "ok",
                "total_duration": 1_000_000,
                "prompt_eval_count": 1024,
                "prompt_eval_duration": 1_000_000_000,
                "eval_count": 4,
                "eval_duration": 500_000_000,
            }

    result = run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        expanded_plan=expanded_plan,
        session_timestamp=datetime(2026, 4, 19, 13, 20, 0, tzinfo=timezone.utc),
    )

    assert call_log.count(1) == 1
    assert call_log.count(64) == 2
    assert [run.metrics["calibration_status"] for run in result.runs] == ["exact", "exact"]
    assert [run.metrics["actual_prompt_tokens"] for run in result.runs] == [1024, 1024]
    assert [run.metrics["calibration_cache_hit"] for run in result.runs] == [False, True]


def test_context_calibration_marks_non_converged_runs_approximate(
    tmp_path: Path,
) -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.CONTEXT_SCALING],
        execution_settings={
            "repetitions": 1,
            "warmup_runs": 1,
            "warmup_enabled": False,
        },
    )
    scenario = build_benchmark_scenarios(BenchmarkType.CONTEXT_SCALING, 4096)[0]
    expanded_plan = [
        PlannedRun(
            run_id="run-context-calibration-failed",
            run_index=1,
            model_name=plan.model_name,
            context_size=4096,
            context_index=1,
            benchmark_type=BenchmarkType.CONTEXT_SCALING,
            benchmark_type_index=1,
            scenario_id=scenario.scenario_id,
            scenario_index=1,
            repetition_index=1,
            scenario_name=scenario.name,
            scenario_version=scenario.version,
        )
    ]
    call_log: list[int] = []

    class FakeClient:
        def list_models(self) -> list[str]:
            return [plan.model_name]

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            call_log.append(int(options["num_predict"]))
            return {
                "response": "ok",
                "total_duration": 1_000_000,
                "prompt_eval_count": 900,
                "prompt_eval_duration": 1_000_000_000,
                "eval_count": 4,
                "eval_duration": 500_000_000,
            }

    result = run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        expanded_plan=expanded_plan,
        session_timestamp=datetime(2026, 4, 19, 13, 25, 0, tzinfo=timezone.utc),
    )

    calibrated_run = result.runs[0]
    assert len(call_log) == 5
    assert call_log[0] == 1
    assert call_log[-1] == 64
    assert calibrated_run.state is RunState.COMPLETED
    assert calibrated_run.metrics["calibration_status"] == "approximate"
    assert calibrated_run.metrics["calibration_attempts"] == 4
    assert calibrated_run.metrics["actual_prompt_tokens"] == 900
    assert calibrated_run.metrics["eligible_for_strict_aggregate"] is True
    assert calibrated_run.metrics["eligible_for_calibrated_context_aggregate"] is True


def test_context_calibration_failure_is_ineligible_for_strict_aggregates(
    tmp_path: Path,
) -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.CONTEXT_SCALING],
        execution_settings={
            "repetitions": 1,
            "warmup_runs": 1,
            "warmup_enabled": False,
        },
    )
    scenario = build_benchmark_scenarios(BenchmarkType.CONTEXT_SCALING, 4096)[0]
    expanded_plan = [
        PlannedRun(
            run_id="run-context-calibration-invalid",
            run_index=1,
            model_name=plan.model_name,
            context_size=4096,
            context_index=1,
            benchmark_type=BenchmarkType.CONTEXT_SCALING,
            benchmark_type_index=1,
            scenario_id=scenario.scenario_id,
            scenario_index=1,
            repetition_index=1,
            scenario_name=scenario.name,
            scenario_version=scenario.version,
        )
    ]
    call_log: list[int] = []

    class FakeClient:
        def list_models(self) -> list[str]:
            return [plan.model_name]

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            call_log.append(int(options["num_predict"]))
            return {
                "response": "ok",
                "total_duration": 1_000_000,
                "prompt_eval_count": None,
                "prompt_eval_duration": 1_000_000_000,
                "eval_count": 4,
                "eval_duration": 500_000_000,
            }

    result = run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        expanded_plan=expanded_plan,
        session_timestamp=datetime(2026, 4, 19, 13, 30, 0, tzinfo=timezone.utc),
    )

    calibrated_run = result.runs[0]
    assert call_log == [1, 64]
    assert calibrated_run.metrics["calibration_status"] == "failed"
    assert calibrated_run.metrics["calibration_attempts"] == 1
    assert calibrated_run.metrics["actual_prompt_tokens"] is None
    assert calibrated_run.metrics["eligible_for_strict_aggregate"] is False
    assert calibrated_run.metrics["eligible_for_calibrated_context_aggregate"] is False


def test_run_profile_session_warms_once_per_model_context_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[4096, 8192],
        benchmark_types=[BenchmarkType.SMOKE],
        execution_settings={
            "repetitions": 2,
            "warmup_runs": 2,
            "warmup_enabled": True,
        },
    )
    call_log: list[str] = []

    class FakeRunner:
        def __init__(self, **_: object) -> None:
            return None

        def run(self, planned_run: PlannedRun) -> RunResult:
            call_log.append(f"run:{planned_run.context_size}:{planned_run.repetition_index}")
            return RunResult(
                run_id=planned_run.run_id,
                run_index=planned_run.run_index,
                model_name=planned_run.model_name,
                context_size=planned_run.context_size,
                context_index=planned_run.context_index,
                benchmark_type=planned_run.benchmark_type,
                benchmark_type_index=planned_run.benchmark_type_index,
                scenario_id=planned_run.scenario_id,
                scenario_index=planned_run.scenario_index,
                scenario_version=planned_run.scenario_version,
                repetition_index=planned_run.repetition_index,
                scenario_name=planned_run.scenario_name,
                state=RunState.COMPLETED,
                elapsed_ms=10.0,
                metrics={"tokens_per_second": 5.0},
            )

    class FakeClient:
        def list_models(self) -> list[str]:
            return [plan.model_name]

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            call_log.append(f"warmup:{options['num_ctx']}")
            return {
                "response": "ok",
                "total_duration": 1_000_000,
                "prompt_eval_count": 1,
                "prompt_eval_duration": 1_000_000,
                "eval_count": 1,
                "eval_duration": 1_000_000,
            }

        def unload_model(self, *, model: str) -> dict[str, object]:
            raise AssertionError("cold prep should not be used for smoke runs")

        def preload_model(self, *, model: str) -> dict[str, object]:
            raise AssertionError("warm prep should not be used for smoke runs")

    monkeypatch.setattr("ollama_workload_profiler.session.BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(
        "ollama_workload_profiler.session._build_host_metadata",
        lambda: {
            "hostname": "bench-host",
            "os": {"platform": "test-os", "release": "1.0"},
            "cpu": {"logical_cores": 16, "physical_cores": 8},
            "memory": {"total_mb": 32768.0},
        },
    )
    monkeypatch.setattr(
        "ollama_workload_profiler.session._build_run_system_snapshot",
        lambda: {
            "cpu_percent": 21.5,
            "memory_available_mb": 24000.0,
            "memory_used_percent": 26.8,
            "ollama_process_count": 2,
        },
    )

    result = run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        session_timestamp=datetime(2026, 4, 19, 12, 15, 0, tzinfo=timezone.utc),
    )

    assert call_log == [
        "warmup:4096",
        "warmup:4096",
        "run:4096:1",
        "run:4096:2",
        "warmup:8192",
        "warmup:8192",
        "run:8192:1",
        "run:8192:2",
    ]
    assert [run.metrics["actual_prep_method"] for run in result.runs] == [
        "session_warmup",
        "already_warm",
        "session_warmup",
        "already_warm",
    ]
    assert all(run.metrics["requested_prep_behavior"] == "session_warmup" for run in result.runs)


def test_run_profile_session_skips_warmup_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE],
        execution_settings={
            "repetitions": 1,
            "warmup_runs": 1,
            "warmup_enabled": False,
        },
    )
    call_log: list[str] = []

    class FakeRunner:
        def __init__(self, **_: object) -> None:
            return None

        def run(self, planned_run: PlannedRun) -> RunResult:
            call_log.append(f"run:{planned_run.context_size}:{planned_run.repetition_index}")
            return RunResult(
                run_id=planned_run.run_id,
                run_index=planned_run.run_index,
                model_name=planned_run.model_name,
                context_size=planned_run.context_size,
                context_index=planned_run.context_index,
                benchmark_type=planned_run.benchmark_type,
                benchmark_type_index=planned_run.benchmark_type_index,
                scenario_id=planned_run.scenario_id,
                scenario_index=planned_run.scenario_index,
                scenario_version=planned_run.scenario_version,
                repetition_index=planned_run.repetition_index,
                scenario_name=planned_run.scenario_name,
                state=RunState.COMPLETED,
                elapsed_ms=10.0,
                metrics={"tokens_per_second": 5.0},
            )

    class FakeClient:
        def list_models(self) -> list[str]:
            return [plan.model_name]

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            raise AssertionError("warmup should be disabled")

    monkeypatch.setattr("ollama_workload_profiler.session.BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(
        "ollama_workload_profiler.session._build_host_metadata",
        lambda: {
            "hostname": "bench-host",
            "os": {"platform": "test-os", "release": "1.0"},
            "cpu": {"logical_cores": 16, "physical_cores": 8},
            "memory": {"total_mb": 32768.0},
        },
    )
    monkeypatch.setattr(
        "ollama_workload_profiler.session._build_run_system_snapshot",
        lambda: {
            "cpu_percent": 21.5,
            "memory_available_mb": 24000.0,
            "memory_used_percent": 26.8,
            "ollama_process_count": 2,
        },
    )

    result = run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        session_timestamp=datetime(2026, 4, 19, 12, 30, 0, tzinfo=timezone.utc),
    )

    assert call_log == ["run:4096:1"]
    assert result.runs[0].metrics["actual_prep_method"] == "warmup_disabled"
    assert result.runs[0].metrics["requested_prep_behavior"] == "session_warmup"


def test_run_profile_session_keeps_explicit_warm_start_separate_from_generic_warmup_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.COLD_WARM, BenchmarkType.SMOKE],
        execution_settings={
            "repetitions": 1,
            "warmup_runs": 1,
            "warmup_enabled": True,
        },
    )
    warm_scenarios = build_benchmark_scenarios(BenchmarkType.COLD_WARM, 4096)
    smoke_scenarios = build_benchmark_scenarios(BenchmarkType.SMOKE, 4096)
    expanded_plan = [
        PlannedRun(
            run_id="run-warm-start",
            run_index=1,
            model_name=plan.model_name,
            context_size=4096,
            context_index=1,
            benchmark_type=BenchmarkType.COLD_WARM,
            benchmark_type_index=1,
            scenario_id=warm_scenarios[1].scenario_id,
            scenario_index=2,
            repetition_index=1,
            scenario_name=warm_scenarios[1].name,
            scenario_version=warm_scenarios[1].version,
        ),
        PlannedRun(
            run_id="run-smoke",
            run_index=2,
            model_name=plan.model_name,
            context_size=4096,
            context_index=1,
            benchmark_type=BenchmarkType.SMOKE,
            benchmark_type_index=1,
            scenario_id=smoke_scenarios[0].scenario_id,
            scenario_index=1,
            repetition_index=1,
            scenario_name=smoke_scenarios[0].name,
            scenario_version=smoke_scenarios[0].version,
        ),
    ]
    call_log: list[str] = []

    class FakeRunner:
        def __init__(self, **_: object) -> None:
            return None

        def run(self, planned_run: PlannedRun) -> RunResult:
            call_log.append(f"run:{planned_run.scenario_id}")
            return RunResult(
                run_id=planned_run.run_id,
                run_index=planned_run.run_index,
                model_name=planned_run.model_name,
                context_size=planned_run.context_size,
                context_index=planned_run.context_index,
                benchmark_type=planned_run.benchmark_type,
                benchmark_type_index=planned_run.benchmark_type_index,
                scenario_id=planned_run.scenario_id,
                scenario_index=planned_run.scenario_index,
                scenario_version=planned_run.scenario_version,
                repetition_index=planned_run.repetition_index,
                scenario_name=planned_run.scenario_name,
                state=RunState.COMPLETED,
                elapsed_ms=10.0,
                metrics={"tokens_per_second": 5.0},
            )

    class FakeClient:
        def list_models(self) -> list[str]:
            call_log.append("list_models")
            return [plan.model_name]

        def preload_model(self, *, model: str, options: dict[str, object] | None = None) -> dict[str, object]:
            assert options == {
                "num_ctx": 4096,
                "num_predict": 16,
                "seed": 42,
                "temperature": 0.0,
            }
            call_log.append(f"preload:{model}")
            return {"done": True}

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            call_log.append(f"warmup:{options['num_ctx']}")
            return {
                "response": "ok",
                "total_duration": 1_000_000,
                "prompt_eval_count": 1,
                "prompt_eval_duration": 1_000_000,
                "eval_count": 1,
                "eval_duration": 1_000_000,
            }

    monkeypatch.setattr("ollama_workload_profiler.session.BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(
        "ollama_workload_profiler.session._build_host_metadata",
        lambda: {
            "hostname": "bench-host",
            "os": {"platform": "test-os", "release": "1.0"},
            "cpu": {"logical_cores": 16, "physical_cores": 8},
            "memory": {"total_mb": 32768.0},
        },
    )
    monkeypatch.setattr(
        "ollama_workload_profiler.session._build_run_system_snapshot",
        lambda: {
            "cpu_percent": 21.5,
            "memory_available_mb": 24000.0,
            "memory_used_percent": 26.8,
            "ollama_process_count": 2,
        },
    )

    result = run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        expanded_plan=expanded_plan,
        session_timestamp=datetime(2026, 4, 19, 12, 45, 0, tzinfo=timezone.utc),
    )

    assert call_log == [
        "list_models",
        "preload:llama3.2",
        "run:cold-warm-warm-start-v1",
        "warmup:4096",
        "run:smoke-basic-v1",
    ]
    assert result.runs[0].metrics["actual_prep_method"] == "explicit_preload"
    assert result.runs[1].metrics["actual_prep_method"] == "session_warmup"


def test_run_profile_session_marks_failed_enforced_cold_sample_ineligible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.COLD_WARM],
        execution_settings={
            "repetitions": 1,
            "warmup_runs": 1,
            "warmup_enabled": True,
        },
    )

    class FakeRunner:
        def __init__(self, **_: object) -> None:
            return None

        def run(self, planned_run: PlannedRun) -> RunResult:
            return RunResult(
                run_id=planned_run.run_id,
                run_index=planned_run.run_index,
                model_name=planned_run.model_name,
                context_size=planned_run.context_size,
                context_index=planned_run.context_index,
                benchmark_type=planned_run.benchmark_type,
                benchmark_type_index=planned_run.benchmark_type_index,
                scenario_id=planned_run.scenario_id,
                scenario_index=planned_run.scenario_index,
                scenario_version=planned_run.scenario_version,
                repetition_index=planned_run.repetition_index,
                scenario_name=planned_run.scenario_name,
                state=RunState.COMPLETED,
                elapsed_ms=10.0,
                metrics={"tokens_per_second": 5.0},
            )

    class FakeClient:
        def list_models(self) -> list[str]:
            return [plan.model_name]

        def unload_model(self, *, model: str) -> dict[str, object]:
            raise RuntimeError("unload failed")

    monkeypatch.setattr("ollama_workload_profiler.session.BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(
        "ollama_workload_profiler.session._build_host_metadata",
        lambda: {
            "hostname": "bench-host",
            "os": {"platform": "test-os", "release": "1.0"},
            "cpu": {"logical_cores": 16, "physical_cores": 8},
            "memory": {"total_mb": 32768.0},
        },
    )
    monkeypatch.setattr(
        "ollama_workload_profiler.session._build_run_system_snapshot",
        lambda: {
            "cpu_percent": 21.5,
            "memory_available_mb": 24000.0,
            "memory_used_percent": 26.8,
            "ollama_process_count": 2,
        },
    )

    result = run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        session_timestamp=datetime(2026, 4, 19, 13, 0, 0, tzinfo=timezone.utc),
    )

    cold_run = result.runs[0]
    assert cold_run.metrics["requested_prep_behavior"] == "cold_start"
    assert cold_run.metrics["actual_prep_method"] == "explicit_unload_failed"
    assert cold_run.metrics["prep_enforcement_succeeded"] is False
    assert cold_run.metrics["eligible_for_strict_aggregate"] is False
    assert cold_run.metrics["eligible_for_cold_start_aggregate"] is False


def test_run_profile_session_warms_multi_turn_chat_once_per_context_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[4096, 8192],
        benchmark_types=[BenchmarkType.USE_CASE_PROFILES],
        execution_settings={
            "repetitions": 1,
            "warmup_runs": 1,
            "warmup_enabled": True,
        },
    )
    scenarios_4096 = build_benchmark_scenarios(BenchmarkType.USE_CASE_PROFILES, 4096)
    scenarios_8192 = build_benchmark_scenarios(BenchmarkType.USE_CASE_PROFILES, 8192)
    chat_4096 = next(
        scenario for scenario in scenarios_4096 if scenario.scenario_id == "use-case-profiles-multi-turn-chat-v1"
    )
    chat_8192 = next(
        scenario for scenario in scenarios_8192 if scenario.scenario_id == "use-case-profiles-multi-turn-chat-v1"
    )
    expanded_plan = [
        PlannedRun(
            run_id="run-chat-4096",
            run_index=1,
            model_name=plan.model_name,
            context_size=4096,
            context_index=1,
            benchmark_type=BenchmarkType.USE_CASE_PROFILES,
            benchmark_type_index=1,
            scenario_id=chat_4096.scenario_id,
            scenario_index=3,
            repetition_index=1,
            scenario_name=chat_4096.name,
            scenario_version=chat_4096.version,
        ),
        PlannedRun(
            run_id="run-chat-8192",
            run_index=2,
            model_name=plan.model_name,
            context_size=8192,
            context_index=2,
            benchmark_type=BenchmarkType.USE_CASE_PROFILES,
            benchmark_type_index=1,
            scenario_id=chat_8192.scenario_id,
            scenario_index=3,
            repetition_index=1,
            scenario_name=chat_8192.name,
            scenario_version=chat_8192.version,
        ),
    ]
    call_log: list[str] = []

    class FakeRunner:
        def __init__(self, **_: object) -> None:
            return None

        def run(self, planned_run: PlannedRun) -> RunResult:
            call_log.append(f"run:{planned_run.context_size}")
            return RunResult(
                run_id=planned_run.run_id,
                run_index=planned_run.run_index,
                model_name=planned_run.model_name,
                context_size=planned_run.context_size,
                context_index=planned_run.context_index,
                benchmark_type=planned_run.benchmark_type,
                benchmark_type_index=planned_run.benchmark_type_index,
                scenario_id=planned_run.scenario_id,
                scenario_index=planned_run.scenario_index,
                scenario_version=planned_run.scenario_version,
                repetition_index=planned_run.repetition_index,
                scenario_name=planned_run.scenario_name,
                state=RunState.COMPLETED,
                elapsed_ms=10.0,
                metrics={"tokens_per_second": 5.0},
            )

    class FakeClient:
        def list_models(self) -> list[str]:
            return [plan.model_name]

        def chat(self, *, model: str, messages: list[dict[str, object]], options: dict[str, object]) -> dict[str, object]:
            assert model == plan.model_name
            call_log.append(f"warmup-chat:{options['num_ctx']}")
            assert [message["content"] for message in messages] == list(MULTI_TURN_CHAT_TURNS)
            return {
                "message": {"content": "ok"},
                "total_duration": 1_000_000,
                "prompt_eval_count": 1,
                "prompt_eval_duration": 1_000_000,
                "eval_count": 1,
                "eval_duration": 1_000_000,
            }

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            raise AssertionError("multi-turn warmup should use chat, not generate")

    monkeypatch.setattr("ollama_workload_profiler.session.BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(
        "ollama_workload_profiler.session._build_host_metadata",
        lambda: {
            "hostname": "bench-host",
            "os": {"platform": "test-os", "release": "1.0"},
            "cpu": {"logical_cores": 16, "physical_cores": 8},
            "memory": {"total_mb": 32768.0},
        },
    )
    monkeypatch.setattr(
        "ollama_workload_profiler.session._build_run_system_snapshot",
        lambda: {
            "cpu_percent": 21.5,
            "memory_available_mb": 24000.0,
            "memory_used_percent": 26.8,
            "ollama_process_count": 2,
        },
    )

    result = run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        expanded_plan=expanded_plan,
        session_timestamp=datetime(2026, 4, 19, 13, 5, 0, tzinfo=timezone.utc),
    )

    assert call_log == [
        "warmup-chat:4096",
        "run:4096",
        "warmup-chat:8192",
        "run:8192",
    ]
    assert [run.metrics["requested_prep_behavior"] for run in result.runs] == [
        "session_warmup",
        "session_warmup",
    ]
    assert [run.metrics["actual_prep_method"] for run in result.runs] == [
        "session_warmup",
        "session_warmup",
    ]
