from __future__ import annotations

import copy
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import tomllib

import pytest

from ollama_workload_profiler.models.results import RunResult, RunState
from ollama_workload_profiler.models.failures import FailureInfo
from ollama_workload_profiler.models.summary import ReportSummary
from ollama_workload_profiler.reporting.artifacts import write_session_artifacts
from ollama_workload_profiler.reporting.markdown import render_markdown_report
from ollama_workload_profiler.reporting.summary import build_report_summary


def test_write_session_artifacts_creates_fixed_filenames(tmp_path: Path) -> None:
    summary = build_report_summary(plan={"model_name": "llama3.2"}, environment={}, runs=[])
    report_markdown = render_markdown_report(summary)

    output_dir = write_session_artifacts(
        tmp_path,
        session_timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
        plan={"model_name": "llama3.2"},
        expanded_plan=[],
        environment={},
        runs=[],
        summary=summary,
        report_markdown=report_markdown,
    )

    assert output_dir.name == "session-2026-04-18_T12-00-00Z"
    assert (output_dir / "plan.json").exists()
    assert (output_dir / "expanded_plan.json").exists()
    assert (output_dir / "environment.json").exists()
    assert (output_dir / "raw.jsonl").exists()
    assert (output_dir / "raw.csv").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()


def test_write_session_artifacts_uses_raw_jsonl_as_source_of_truth_for_raw_csv(tmp_path: Path) -> None:
    runs = [
        RunResult(
            run_id="run-0001",
            run_index=1,
            model_name="llama3.2",
            context_size=4096,
            context_index=1,
            benchmark_type="smoke",
            benchmark_type_index=1,
            scenario_id="smoke-basic-v1",
            scenario_index=1,
            state=RunState.COMPLETED,
            elapsed_ms=12.5,
            system_snapshot={
                "cpu_percent": 21.5,
                "memory_available_mb": 24000.0,
                "ollama_process_count": 2,
            },
            metrics={
                "tokens_per_second": 45.0,
                "phase_peaks": {"generation": {"rss_mb": 256.0, "cpu_percent": 65.0}},
            },
        )
    ]
    summary = build_report_summary(plan={"model_name": "llama3.2"}, environment={}, runs=runs)

    output_dir = write_session_artifacts(
        tmp_path,
        session_timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
        plan={"model_name": "llama3.2"},
        expanded_plan=[],
        environment={"platform": "test"},
        runs=runs,
        summary=summary,
        report_markdown=render_markdown_report(summary),
    )

    raw_lines = (output_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 1

    raw_payload = json.loads(raw_lines[0])
    assert raw_payload["run_id"] == "run-0001"
    assert raw_payload["metrics"]["phase_peaks"]["generation"]["rss_mb"] == 256.0

    with (output_dir / "raw.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert len(csv_rows) == 1
    assert csv_rows[0]["run_id"] == raw_payload["run_id"]
    assert csv_rows[0]["metrics.tokens_per_second"] == "45.0"
    assert csv_rows[0]["metrics.phase_peaks.generation.rss_mb"] == "256.0"
    assert csv_rows[0]["system_snapshot.cpu_percent"] == "21.5"
    assert csv_rows[0]["system_snapshot.memory_available_mb"] == "24000.0"


def test_write_session_artifacts_creates_unique_session_directories_without_overwriting(tmp_path: Path) -> None:
    summary = build_report_summary(plan={"model_name": "llama3.2"}, environment={}, runs=[])

    first_output_dir = write_session_artifacts(
        tmp_path,
        session_timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
        plan={"model_name": "llama3.2"},
        expanded_plan=[],
        environment={},
        runs=[],
        summary=summary,
        report_markdown=render_markdown_report(summary),
    )
    second_output_dir = write_session_artifacts(
        tmp_path,
        session_timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
        plan={"model_name": "llama3.2"},
        expanded_plan=[],
        environment={},
        runs=[],
        summary=summary,
        report_markdown=render_markdown_report(summary),
    )

    assert first_output_dir.name == "session-2026-04-18_T12-00-00Z"
    assert second_output_dir.name == "session-2026-04-18_T12-00-00Z-01"
    assert first_output_dir != second_output_dir


def test_write_session_artifacts_rejects_invalid_base_directory(tmp_path: Path) -> None:
    invalid_base_dir = tmp_path / "not-a-directory.txt"
    invalid_base_dir.write_text("x", encoding="utf-8")
    summary = build_report_summary(plan={"model_name": "llama3.2"}, environment={}, runs=[])

    with pytest.raises(ValueError, match="base_dir must be an existing directory"):
        write_session_artifacts(
            invalid_base_dir,
            session_timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
            plan={"model_name": "llama3.2"},
            expanded_plan=[],
            environment={},
            runs=[],
            summary=summary,
            report_markdown=render_markdown_report(summary),
        )


def test_build_report_summary_is_aggregate_level_and_does_not_mutate_run_data() -> None:
    runs = [
        RunResult(
            run_id="run-0001",
            run_index=1,
            model_name="llama3.2",
            context_size=4096,
            context_index=1,
            benchmark_type="smoke",
            benchmark_type_index=1,
            scenario_id="smoke-basic-v1",
            scenario_index=1,
            state=RunState.COMPLETED,
            elapsed_ms=12.5,
            metrics={
                "tokens_per_second": 45.0,
                "phase_peaks": {"generation": {"rss_mb": 256.0, "cpu_percent": 65.0}},
            },
        ),
        RunResult(
            run_id="run-0002",
            run_index=2,
            model_name="llama3.2",
            context_size=4096,
            context_index=1,
            benchmark_type="ttft",
            benchmark_type_index=2,
            scenario_id="ttft-basic-v1",
            scenario_index=1,
            state=RunState.FAILED,
            metrics={"finalization_error": "sampler shutdown failed"},
        ),
    ]
    original_runs = [copy.deepcopy(run.model_dump(mode="json")) for run in runs]

    summary = build_report_summary(
        plan={"model_name": "llama3.2"},
        environment={"platform": "test"},
        runs=runs,
    )
    payload = summary.model_dump(mode="json")

    assert isinstance(summary, ReportSummary)
    assert payload["session_metrics"]["run_count"] == 2
    assert payload["session_metrics"]["completed_runs"] == 1
    assert payload["session_metrics"]["failed_runs"] == 1
    assert "runs" not in payload
    assert payload["benchmark_summaries"][0]["sample_size"] >= 1
    assert payload["phase_peak_summaries"][0]["phase"] == "generation"
    assert payload["failures"][0]["message"] == "sampler shutdown failed"
    assert [run.model_dump(mode="json") for run in runs] == original_runs


def test_build_report_summary_uses_recorded_runs_for_planned_run_count() -> None:
    runs = [
        RunResult(
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
            state=RunState.COMPLETED,
        ),
        RunResult(
            run_id="run-0002",
            run_index=2,
            model_name="llama3.2",
            context_size=4096,
            context_index=1,
            benchmark_type="use-case-profiles",
            benchmark_type_index=2,
            scenario_id="use-case-profiles-quick-qa-v1",
            scenario_index=1,
            scenario_version="v1",
            state=RunState.COMPLETED,
        ),
    ]

    summary = build_report_summary(
        plan={
            "model_name": "llama3.2",
            "contexts": [4096, 8192],
            "benchmark_types": ["smoke", "use-case-profiles"],
        },
        environment={"platform": "test"},
        runs=runs,
    )

    assert summary.session_metrics["planned_run_count"] == 2


def test_build_report_summary_prefers_explicit_failure_payloads() -> None:
    summary = build_report_summary(
        plan={"model_name": "llama3.2"},
        environment={"platform": "test"},
        runs=[
            RunResult(
                run_id="run-0001",
                run_index=1,
                model_name="llama3.2",
                context_size=4096,
                context_index=1,
                benchmark_type="ttft",
                benchmark_type_index=1,
                scenario_id="ttft-basic-v1",
                scenario_index=1,
                scenario_version="v1",
                state=RunState.FAILED,
                failure=FailureInfo(
                    kind="failed",
                    phase="finalization",
                    message="sampler shutdown failed",
                    exception_class="RuntimeError",
                ),
                metrics={"finalization_error": "sampler shutdown failed"},
            )
        ],
    )

    assert summary.failures[0].phase == "finalization"
    assert summary.failures[0].exception_class == "RuntimeError"


def test_render_markdown_report_contains_required_sections_and_handles_minimal_data() -> None:
    summary = build_report_summary(plan={"model_name": "llama3.2"}, environment={}, runs=[])

    markdown = render_markdown_report(summary)

    assert "# Benchmark Report" in markdown
    assert "## Executive summary" in markdown
    assert "## Per-model summary" in markdown
    assert "## Use-case suitability matrix" in markdown
    assert "## Detailed timing and resource tables" in markdown
    assert "## Phase-level resource peaks" in markdown
    assert "## Warnings and failures" in markdown
    assert "## Plain-language recommendations" in markdown
    assert "No completed runs were recorded." in markdown
    assert "Artifacts:" not in markdown


def test_render_markdown_report_uses_only_the_required_section_contract() -> None:
    summary = build_report_summary(plan={"model_name": "llama3.2"}, environment={}, runs=[])

    markdown = render_markdown_report(summary)

    headings = [line for line in markdown.splitlines() if line.startswith("## ")]
    assert headings == [
        "## Executive summary",
        "## Per-model summary",
        "## Use-case suitability matrix",
        "## Detailed timing and resource tables",
        "## Phase-level resource peaks",
        "## Warnings and failures",
        "## Plain-language recommendations",
    ]
    assert "Artifacts:" not in markdown


def test_render_markdown_report_renders_use_case_cells_readably() -> None:
    summary = build_report_summary(
        plan={
            "model_name": "llama3.2",
            "contexts": [4096],
            "benchmark_types": ["use-case-profiles"],
        },
        environment={"platform": "test"},
        runs=[
            RunResult(
                run_id="run-0001",
                run_index=1,
                model_name="llama3.2",
                context_size=4096,
                context_index=1,
                benchmark_type="use-case-profiles",
                benchmark_type_index=1,
                scenario_id="use-case-profiles-quick-qa-v1",
                scenario_index=1,
                scenario_version="v1",
                scenario_name="Quick QA",
                state=RunState.COMPLETED,
                elapsed_ms=910.0,
                metrics={"tokens_per_second": 22.0},
            )
        ],
    )

    markdown = render_markdown_report(summary)

    assert "Good fit with caution" in markdown
    assert "UseCaseMatrixCell(" not in markdown


def test_pinned_requirements_match_pyproject_dependencies() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project_dependencies = pyproject["project"]["dependencies"]
    requirements = [
        line.strip()
        for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert requirements == project_dependencies
