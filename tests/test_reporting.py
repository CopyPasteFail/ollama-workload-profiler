from __future__ import annotations

import copy
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import tomllib

import pytest

from ollama_workload_profiler.methodology import BENCHMARK_METHODOLOGY_VERSION
from ollama_workload_profiler.models.results import RunResult, RunState
from ollama_workload_profiler.models.plan import BenchmarkSessionPlan, BenchmarkType
from ollama_workload_profiler.models.failures import FailureInfo
from ollama_workload_profiler.models.summary import ReportSummary
from ollama_workload_profiler.reporting.artifacts import write_session_artifacts
from ollama_workload_profiler.reporting.markdown import render_markdown_report
from ollama_workload_profiler.reporting.plots import export_summary_plots
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


def test_export_summary_plots_writes_blog_friendly_svg_artifacts(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "summary.json").write_text(
        json.dumps(
            {
                "benchmark_summaries": [
                    {
                        "model_name": "llama3.2",
                        "context_size": 4096,
                        "benchmark_type": "prompt-scaling",
                        "scenario_id": "prompt-scaling-small-v1",
                        "strict_sample_size": 1,
                        "prompt_eval_count_median": 1024.0,
                        "ttft_ms_median": 80.0,
                        "stream_emission_interval_ms_median_median": 18.0,
                        "prompt_tokens_per_second_median": 2100.0,
                        "generation_tokens_per_second_median": 52.0,
                    },
                    {
                        "model_name": "llama3.2",
                        "context_size": 4096,
                        "benchmark_type": "prompt-scaling",
                        "scenario_id": "prompt-scaling-large-v1",
                        "strict_sample_size": 1,
                        "prompt_eval_count_median": 4096.0,
                        "ttft_ms_median": 140.0,
                        "stream_emission_interval_ms_median_median": 26.0,
                        "prompt_tokens_per_second_median": 1800.0,
                        "generation_tokens_per_second_median": 48.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    written = export_summary_plots(session_dir)

    assert [path.name for path in written] == [
        "latency_vs_prompt_size.svg",
        "throughput_vs_prompt_size.svg",
        "README.md",
    ]
    latency_svg = (session_dir / "plots" / "latency_vs_prompt_size.svg").read_text(encoding="utf-8")
    throughput_svg = (session_dir / "plots" / "throughput_vs_prompt_size.svg").read_text(encoding="utf-8")
    note = (session_dir / "plots" / "README.md").read_text(encoding="utf-8")
    assert "TTFT vs prompt size" in latency_svg
    assert "Median stream emission interval" in latency_svg
    assert "Prompt processing speed vs prompt size" in throughput_svg
    assert "Generation speed" in throughput_svg
    assert "prompt_eval_count_median" in note


def test_write_session_artifacts_persists_execution_settings_as_requested_policy(tmp_path: Path) -> None:
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
    summary = build_report_summary(plan=plan, environment={}, runs=[])

    output_dir = write_session_artifacts(
        tmp_path,
        session_timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc),
        plan=plan,
        expanded_plan=[],
        environment={},
        runs=[],
        summary=summary,
        report_markdown=render_markdown_report(summary),
    )

    plan_payload = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
    assert plan_payload["execution_settings"]["repetitions"] == 3
    assert plan_payload["execution_settings"]["seed"] == 1234
    assert plan_payload["execution_settings"]["temperature"] == 0.2
    assert plan_payload["execution_settings"]["top_p"] == 0.9
    assert plan_payload["execution_settings"]["warmup_runs"] == 1
    assert plan_payload["execution_settings"]["warmup_enabled"] is True
    assert "repetitions" not in plan_payload


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
                "system_cpu_load_snapshot": 21.5,
                "memory_available_mb": 24000.0,
                "available_system_memory_mb": 24000.0,
                "host_pressure_warning": True,
                "host_pressure_warning_reasons": ["system_cpu_load_snapshot >= 80%"],
                "ollama_process_count": 2,
            },
            metrics={
                "tokens_per_second": 45.0,
                "stream_emission_count": 3,
                "stream_emission_offsets_ms": [12.0, 42.0, 92.0],
                "stream_duration_ms": 80.0,
                "stream_emission_interval_ms_median": 40.0,
                "stream_emission_interval_ms_p95": 50.0,
                "stream_output_units_per_second": 37.5,
                "stream_output_unit": "emission",
                "concurrency_parallelism": 2,
                "concurrency_request_count": 2,
                "concurrency_request_elapsed_ms_p50": 125.0,
                "concurrency_request_elapsed_ms_p95": 150.0,
                "concurrency_request_ttft_ms_p50": 20.0,
                "concurrency_request_ttft_ms_p95": 25.0,
                "concurrency_requests": [
                    {"request_index": 1, "elapsed_ms": 100.0, "ttft_ms": 15.0},
                    {"request_index": 2, "elapsed_ms": 150.0, "ttft_ms": 25.0},
                ],
                "gpu_telemetry_available": True,
                "gpu_telemetry_source": "nvidia-smi",
                "gpu_backend": "nvidia-smi",
                "peak_gpu_memory_mb": 3072.0,
                "avg_gpu_util_percent": 45.0,
                "peak_gpu_util_percent": 70.0,
                "gpu_device_count": 2,
                "gpu_telemetry_notes": ["multi_gpu_memory_summed_util_averaged"],
                "sampled_process_count": 3,
                "sampled_process_ids": [10, 20, 30],
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
    assert raw_payload["metrics"]["stream_emission_offsets_ms"] == [12.0, 42.0, 92.0]
    assert raw_payload["metrics"]["stream_output_unit"] == "emission"
    assert raw_payload["metrics"]["concurrency_request_elapsed_ms_p50"] == 125.0
    assert raw_payload["metrics"]["concurrency_requests"][0]["elapsed_ms"] == 100.0
    assert raw_payload["metrics"]["gpu_telemetry_available"] is True
    assert raw_payload["metrics"]["gpu_telemetry_source"] == "nvidia-smi"
    assert raw_payload["metrics"]["gpu_telemetry_notes"] == ["multi_gpu_memory_summed_util_averaged"]
    assert raw_payload["metrics"]["sampled_process_count"] == 3
    assert raw_payload["metrics"]["sampled_process_ids"] == [10, 20, 30]
    assert raw_payload["metrics"]["phase_peaks"]["generation"]["rss_mb"] == 256.0
    assert raw_payload["system_snapshot"]["available_system_memory_mb"] == 24000.0
    assert raw_payload["system_snapshot"]["system_cpu_load_snapshot"] == 21.5
    assert raw_payload["system_snapshot"]["host_pressure_warning"] is True

    with (output_dir / "raw.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert len(csv_rows) == 1
    assert csv_rows[0]["run_id"] == raw_payload["run_id"]
    assert csv_rows[0]["metrics.tokens_per_second"] == "45.0"
    assert csv_rows[0]["metrics.stream_emission_count"] == "3"
    assert csv_rows[0]["metrics.stream_emission_offsets_ms"] == "[12.0, 42.0, 92.0]"
    assert csv_rows[0]["metrics.stream_output_units_per_second"] == "37.5"
    assert csv_rows[0]["metrics.stream_output_unit"] == "emission"
    assert csv_rows[0]["metrics.concurrency_request_elapsed_ms_p50"] == "125.0"
    assert csv_rows[0]["metrics.concurrency_request_elapsed_ms_p95"] == "150.0"
    assert '"elapsed_ms": 100.0' in csv_rows[0]["metrics.concurrency_requests"]
    assert csv_rows[0]["metrics.gpu_telemetry_available"] == "True"
    assert csv_rows[0]["metrics.gpu_telemetry_source"] == "nvidia-smi"
    assert csv_rows[0]["metrics.peak_gpu_memory_mb"] == "3072.0"
    assert csv_rows[0]["metrics.avg_gpu_util_percent"] == "45.0"
    assert csv_rows[0]["metrics.peak_gpu_util_percent"] == "70.0"
    assert csv_rows[0]["metrics.sampled_process_count"] == "3"
    assert csv_rows[0]["metrics.sampled_process_ids"] == "[10, 20, 30]"
    assert csv_rows[0]["metrics.phase_peaks.generation.rss_mb"] == "256.0"
    assert csv_rows[0]["system_snapshot.cpu_percent"] == "21.5"
    assert csv_rows[0]["system_snapshot.memory_available_mb"] == "24000.0"
    assert csv_rows[0]["system_snapshot.available_system_memory_mb"] == "24000.0"
    assert csv_rows[0]["system_snapshot.system_cpu_load_snapshot"] == "21.5"
    assert csv_rows[0]["system_snapshot.host_pressure_warning"] == "True"


def test_report_summary_surfaces_host_pressure_context_and_warnings() -> None:
    runs = [
        RunResult(
            run_id="run-0001",
            run_index=1,
            model_name="llama3.2",
            context_size=4096,
            context_index=1,
            benchmark_type=BenchmarkType.SMOKE,
            benchmark_type_index=1,
            scenario_id="smoke-basic-v1",
            scenario_index=1,
            state=RunState.COMPLETED,
            elapsed_ms=12.5,
            system_snapshot={
                "available_system_memory_mb": 768.0,
                "system_cpu_load_snapshot": 87.5,
                "host_pressure_warning": True,
                "host_pressure_warning_reasons": [
                    "system_cpu_load_snapshot >= 80%",
                    "available_system_memory_mb < 1024",
                ],
            },
            metrics={"tokens_per_second": 45.0},
        )
    ]

    summary = build_report_summary(plan={"model_name": "llama3.2"}, environment={}, runs=runs)
    markdown = render_markdown_report(summary)

    assert summary.session_metrics["available_system_memory_mb_median"] == 768.0
    assert summary.session_metrics["system_cpu_load_snapshot_median"] == 87.5
    assert "One or more runs started with advisory host pressure warnings." in summary.warnings
    assert "Host pressure reasons observed: available_system_memory_mb < 1024; system_cpu_load_snapshot >= 80%" in summary.warnings
    assert "- available_system_memory_mb: median 768.0 | p95 768.0 | n=1" not in markdown
    assert "- system_cpu_load_snapshot: median 87.5 | p95 87.5 | n=1" not in markdown


def test_report_summary_surfaces_concurrency_smoke_metrics() -> None:
    runs = [
        RunResult(
            run_id="run-concurrency-smoke",
            run_index=1,
            model_name="llama3.2",
            context_size=4096,
            context_index=1,
            benchmark_type=BenchmarkType.CONCURRENCY_SMOKE,
            benchmark_type_index=1,
            scenario_id="concurrency-smoke-p2-v1",
            scenario_index=1,
            state=RunState.COMPLETED,
            elapsed_ms=160.0,
            metrics={
                "concurrency_parallelism": 2,
                "concurrency_request_elapsed_ms_p50": 125.0,
                "concurrency_request_elapsed_ms_p95": 150.0,
                "concurrency_request_ttft_ms_p50": 20.0,
                "concurrency_request_ttft_ms_p95": 25.0,
            },
        )
    ]

    summary = build_report_summary(plan={"model_name": "llama3.2"}, environment={}, runs=runs)
    markdown = render_markdown_report(summary)

    assert summary.session_metrics["concurrency_request_elapsed_ms_p50_median"] == 125.0
    assert summary.session_metrics["concurrency_request_elapsed_ms_p95_median"] == 150.0
    assert summary.session_metrics["concurrency_request_ttft_ms_p50_median"] == 20.0
    assert summary.session_metrics["concurrency_request_ttft_ms_p95_median"] == 25.0
    assert "- concurrency_request_elapsed_ms_p50: median 125.0 | p95 125.0 | n=1" not in markdown


def test_write_session_artifacts_keep_prep_and_calibration_contract_consistent_across_outputs(
    tmp_path: Path,
) -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.CONTEXT_SCALING],
        execution_settings={
            "repetitions": 2,
            "seed": 1234,
            "temperature": 0.0,
            "warmup_runs": 1,
            "warmup_enabled": False,
        },
    )
    runs = [
        RunResult(
            run_id="run-0001",
            run_index=1,
            model_name="llama3.2",
            context_size=4096,
            context_index=1,
            benchmark_type=BenchmarkType.CONTEXT_SCALING,
            benchmark_type_index=1,
            scenario_id="context-scaling-quarter-v1",
            scenario_index=1,
            scenario_version="v1",
            repetition_index=1,
            scenario_name="Quarter context fill",
            state=RunState.COMPLETED,
            elapsed_ms=12.5,
            metrics={
                "requested_prep_behavior": "calibrated_context_fill",
                "actual_prep_method": "calibration",
                "prep_enforcement_succeeded": True,
                "requested_fill_ratio": 0.25,
                "target_prompt_tokens": 1024,
                "actual_prompt_tokens": 1024,
                "calibration_status": "exact",
                "calibration_attempts": 1,
                "calibration_cache_hit": False,
                "eligible_for_strict_aggregate": True,
                "eligible_for_calibrated_context_aggregate": True,
                "tokens_per_second": 45.0,
                "generation_tokens_per_second": 45.0,
            },
        ),
        RunResult(
            run_id="run-0002",
            run_index=2,
            model_name="llama3.2",
            context_size=4096,
            context_index=1,
            benchmark_type=BenchmarkType.CONTEXT_SCALING,
            benchmark_type_index=1,
            scenario_id="context-scaling-quarter-v1",
            scenario_index=1,
            scenario_version="v1",
            repetition_index=2,
            scenario_name="Quarter context fill",
            state=RunState.COMPLETED,
            elapsed_ms=13.5,
            metrics={
                "requested_prep_behavior": "calibrated_context_fill",
                "actual_prep_method": "calibration",
                "prep_enforcement_succeeded": True,
                "requested_fill_ratio": 0.25,
                "target_prompt_tokens": 1024,
                "actual_prompt_tokens": 1024,
                "calibration_status": "exact",
                "calibration_attempts": 1,
                "calibration_cache_hit": True,
                "eligible_for_strict_aggregate": True,
                "eligible_for_calibrated_context_aggregate": True,
                "tokens_per_second": 47.0,
                "generation_tokens_per_second": 47.0,
            },
        ),
    ]
    environment = {
        "session_started_at": "2026-04-19T14:00:00+00:00",
        "execution_settings": dict(plan.execution_settings),
        "host": {"hostname": "bench-host"},
    }
    summary = build_report_summary(plan=plan, environment=environment, runs=runs)
    report_markdown = render_markdown_report(summary)

    output_dir = write_session_artifacts(
        tmp_path,
        session_timestamp=datetime(2026, 4, 19, 14, 0, 0, tzinfo=timezone.utc),
        plan=plan,
        expanded_plan=[],
        environment=environment,
        runs=runs,
        summary=summary,
        report_markdown=report_markdown,
    )

    plan_payload = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
    environment_payload = json.loads((output_dir / "environment.json").read_text(encoding="utf-8"))
    summary_payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    raw_rows = [
        json.loads(line)
        for line in (output_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    with (output_dir / "raw.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    markdown = (output_dir / "report.md").read_text(encoding="utf-8")

    assert plan_payload["execution_settings"] == environment_payload["execution_settings"]
    assert plan_payload["execution_settings"]["warmup_enabled"] is False
    assert [row["metrics"]["calibration_cache_hit"] for row in raw_rows] == [False, True]
    assert [row["metrics"]["requested_prep_behavior"] for row in raw_rows] == [
        "calibrated_context_fill",
        "calibrated_context_fill",
    ]
    assert csv_rows[0]["metrics.calibration_status"] == "exact"
    assert csv_rows[1]["metrics.calibration_cache_hit"] == "True"
    assert csv_rows[0]["metrics.requested_prep_behavior"] == "calibrated_context_fill"
    assert summary_payload["benchmark_summaries"][0]["strict_sample_size"] == 2
    assert summary_payload["benchmark_summaries"][0]["sample_size"] == 2
    assert plan_payload["benchmark_methodology_version"] == BENCHMARK_METHODOLOGY_VERSION
    assert environment_payload["benchmark_methodology_version"] == BENCHMARK_METHODOLOGY_VERSION
    assert summary_payload["benchmark_methodology_version"] == BENCHMARK_METHODOLOGY_VERSION
    assert summary_payload["artifacts"]["report.md"] == "report.md"
    assert f"Benchmark methodology: {BENCHMARK_METHODOLOGY_VERSION}" in markdown
    assert "| strict_sample_size |" in markdown
    assert "generation_tokens_per_second" in markdown
    assert "\n- tokens_per_second:" not in markdown
    assert "| tokens_per_second |" not in markdown


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


def test_build_report_summary_uses_completed_and_eligible_samples_for_aggregates() -> None:
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
            elapsed_ms=10.0,
            metrics={
                "eligible_for_strict_aggregate": True,
                "eligible_for_ttft_aggregate": True,
                "load_duration_ms": 5.0,
                "ttft_ms": 100.0,
                "stream_emission_count": 3,
                "stream_duration_ms": 80.0,
                "stream_emission_interval_ms_median": 30.0,
                "stream_emission_interval_ms_p95": 50.0,
                "stream_output_units_per_second": 37.5,
                "peak_gpu_memory_mb": 3072.0,
                "avg_gpu_util_percent": 45.0,
                "peak_gpu_util_percent": 70.0,
                "prompt_tokens_per_second": 1000.0,
                "generation_tokens_per_second": 10.0,
                "tokens_per_second": 10.0,
            },
        ),
        RunResult(
            run_id="run-0002",
            run_index=2,
            model_name="llama3.2",
            context_size=4096,
            context_index=1,
            benchmark_type="smoke",
            benchmark_type_index=1,
            scenario_id="smoke-basic-v1",
            scenario_index=1,
            state=RunState.COMPLETED,
            elapsed_ms=20.0,
            metrics={
                "eligible_for_strict_aggregate": True,
                "eligible_for_ttft_aggregate": True,
                "load_duration_ms": 7.0,
                "ttft_ms": 200.0,
                "stream_emission_count": 1,
                "stream_duration_ms": 0.0,
                "stream_emission_interval_ms_median": None,
                "stream_emission_interval_ms_p95": None,
                "stream_output_units_per_second": None,
                "peak_gpu_memory_mb": 4096.0,
                "avg_gpu_util_percent": 55.0,
                "peak_gpu_util_percent": 80.0,
                "prompt_tokens_per_second": 2000.0,
                "generation_tokens_per_second": 20.0,
                "tokens_per_second": 20.0,
            },
        ),
        RunResult(
            run_id="run-0003",
            run_index=3,
            model_name="llama3.2",
            context_size=4096,
            context_index=1,
            benchmark_type="smoke",
            benchmark_type_index=1,
            scenario_id="smoke-basic-v1",
            scenario_index=1,
            state=RunState.COMPLETED,
            elapsed_ms=999.0,
            metrics={
                "eligible_for_strict_aggregate": False,
                "eligible_for_ttft_aggregate": False,
                "load_duration_ms": 50.0,
                "ttft_ms": 5000.0,
                "stream_emission_count": 8,
                "stream_duration_ms": 300.0,
                "stream_emission_interval_ms_median": 40.0,
                "stream_emission_interval_ms_p95": 70.0,
                "stream_output_units_per_second": 26.667,
                "prompt_tokens_per_second": 9000.0,
                "generation_tokens_per_second": 90.0,
                "tokens_per_second": 90.0,
            },
        ),
        RunResult(
            run_id="run-0004",
            run_index=4,
            model_name="llama3.2",
            context_size=4096,
            context_index=1,
            benchmark_type="smoke",
            benchmark_type_index=1,
            scenario_id="smoke-basic-v1",
            scenario_index=1,
            state=RunState.FAILED,
            elapsed_ms=2500.0,
            metrics={
                "eligible_for_strict_aggregate": False,
                "eligible_for_ttft_aggregate": False,
                "load_duration_ms": 120.0,
                "ttft_ms": 8000.0,
                "prompt_tokens_per_second": 12000.0,
                "generation_tokens_per_second": 120.0,
                "tokens_per_second": 120.0,
            },
        ),
    ]

    summary = build_report_summary(plan={"model_name": "llama3.2"}, environment={}, runs=runs)

    assert summary.session_metrics["completed_sample_size"] == 3
    assert summary.session_metrics["elapsed_ms_median"] == 20.0
    assert summary.session_metrics["elapsed_ms_p95"] == 999.0
    assert summary.session_metrics["ttft_ms_sample_size"] == 3
    assert summary.session_metrics["ttft_ms_median"] == 200.0
    assert summary.session_metrics["ttft_ms_p95"] == 5000.0
    assert summary.session_metrics["prompt_tokens_per_second_sample_size"] == 3
    assert summary.session_metrics["prompt_tokens_per_second_median"] == 2000.0
    assert summary.session_metrics["prompt_tokens_per_second_p95"] == 9000.0
    assert summary.session_metrics["generation_tokens_per_second_sample_size"] == 3
    assert summary.session_metrics["generation_tokens_per_second_median"] == 20.0
    assert summary.session_metrics["generation_tokens_per_second_p95"] == 90.0
    assert summary.session_metrics["load_duration_ms_sample_size"] == 3
    assert summary.session_metrics["load_duration_ms_median"] == 7.0
    assert summary.session_metrics["load_duration_ms_p95"] == 50.0
    assert summary.session_metrics["stream_emission_count_sample_size"] == 3
    assert summary.session_metrics["stream_emission_count_median"] == 3.0
    assert summary.session_metrics["stream_emission_count_p95"] == 8.0
    assert summary.session_metrics["stream_duration_ms_sample_size"] == 3
    assert summary.session_metrics["stream_duration_ms_median"] == 80.0
    assert summary.session_metrics["stream_duration_ms_p95"] == 300.0
    assert summary.session_metrics["stream_emission_interval_ms_median_sample_size"] == 2
    assert summary.session_metrics["stream_emission_interval_ms_median_median"] == 35.0
    assert summary.session_metrics["stream_emission_interval_ms_median_p95"] == 40.0
    assert summary.session_metrics["stream_emission_interval_ms_p95_sample_size"] == 2
    assert summary.session_metrics["stream_emission_interval_ms_p95_median"] == 60.0
    assert summary.session_metrics["stream_emission_interval_ms_p95_p95"] == 70.0
    assert summary.session_metrics["stream_output_units_per_second_sample_size"] == 2
    assert summary.session_metrics["stream_output_units_per_second_median"] == 32.084
    assert summary.session_metrics["stream_output_units_per_second_p95"] == 37.5
    assert summary.session_metrics["peak_gpu_memory_mb_sample_size"] == 2
    assert summary.session_metrics["peak_gpu_memory_mb_median"] == 3584.0
    assert summary.session_metrics["peak_gpu_memory_mb_p95"] == 4096.0
    assert summary.session_metrics["avg_gpu_util_percent_sample_size"] == 2
    assert summary.session_metrics["avg_gpu_util_percent_median"] == 50.0
    assert summary.session_metrics["avg_gpu_util_percent_p95"] == 55.0
    assert summary.session_metrics["peak_gpu_util_percent_sample_size"] == 2
    assert summary.session_metrics["peak_gpu_util_percent_median"] == 75.0
    assert summary.session_metrics["peak_gpu_util_percent_p95"] == 80.0

    model_summary = summary.model_summaries["llama3.2"]
    assert model_summary.metrics["completed_sample_size"] == 3
    assert model_summary.metrics["generation_tokens_per_second_median"] == 20.0
    assert model_summary.metrics["generation_tokens_per_second_p95"] == 90.0
    assert model_summary.metrics["stream_duration_ms_median"] == 80.0

    benchmark_summary = summary.benchmark_summaries[0]
    assert benchmark_summary["sample_size"] == 4
    assert benchmark_summary["completed_runs"] == 3
    assert benchmark_summary["strict_sample_size"] == 2
    assert benchmark_summary["elapsed_ms_median"] == 15.0
    assert benchmark_summary["elapsed_ms_p95"] == 20.0
    assert benchmark_summary["ttft_ms_sample_size"] == 2
    assert benchmark_summary["ttft_ms_median"] == 150.0
    assert benchmark_summary["ttft_ms_p95"] == 200.0
    assert benchmark_summary["prompt_tokens_per_second_median"] == 1500.0
    assert benchmark_summary["prompt_tokens_per_second_p95"] == 2000.0
    assert benchmark_summary["generation_tokens_per_second_median"] == 15.0
    assert benchmark_summary["generation_tokens_per_second_p95"] == 20.0
    assert benchmark_summary["load_duration_ms_median"] == 6.0
    assert benchmark_summary["load_duration_ms_p95"] == 7.0
    assert benchmark_summary["stream_emission_count_median"] == 2.0
    assert benchmark_summary["stream_duration_ms_median"] == 40.0
    assert benchmark_summary["stream_emission_interval_ms_median_median"] == 30.0
    assert benchmark_summary["stream_emission_interval_ms_p95_median"] == 50.0
    assert benchmark_summary["stream_output_units_per_second_median"] == 37.5
    assert benchmark_summary["peak_gpu_memory_mb_median"] == 3584.0
    assert benchmark_summary["avg_gpu_util_percent_median"] == 50.0
    assert benchmark_summary["peak_gpu_util_percent_median"] == 75.0


def test_build_report_summary_excludes_failed_prep_and_failed_calibration_from_strict_aggregates() -> None:
    runs = [
        RunResult(
            run_id="run-cold-failed-prep",
            run_index=1,
            model_name="llama3.2",
            context_size=4096,
            context_index=1,
            benchmark_type=BenchmarkType.COLD_WARM,
            benchmark_type_index=1,
            scenario_id="cold-warm-cold-start-v1",
            scenario_index=1,
            scenario_version="v1",
            state=RunState.COMPLETED,
            elapsed_ms=999.0,
            metrics={
                "requested_prep_behavior": "cold_start",
                "actual_prep_method": "explicit_unload_failed",
                "prep_enforcement_succeeded": False,
                "eligible_for_strict_aggregate": False,
                "eligible_for_cold_start_aggregate": False,
                "ttft_ms": 9999.0,
                "load_duration_ms": 99.0,
                "generation_tokens_per_second": 1.0,
                "tokens_per_second": 1.0,
            },
        ),
        RunResult(
            run_id="run-context-failed-calibration",
            run_index=2,
            model_name="llama3.2",
            context_size=4096,
            context_index=1,
            benchmark_type=BenchmarkType.CONTEXT_SCALING,
            benchmark_type_index=2,
            scenario_id="context-scaling-quarter-v1",
            scenario_index=1,
            scenario_version="v1",
            state=RunState.COMPLETED,
            elapsed_ms=888.0,
            metrics={
                "requested_prep_behavior": "calibrated_context_fill",
                "actual_prep_method": "calibration",
                "prep_enforcement_succeeded": True,
                "calibration_status": "failed",
                "eligible_for_strict_aggregate": False,
                "eligible_for_calibrated_context_aggregate": False,
                "ttft_ms": 8888.0,
                "load_duration_ms": 88.0,
                "generation_tokens_per_second": 2.0,
                "tokens_per_second": 2.0,
            },
        ),
        RunResult(
            run_id="run-eligible",
            run_index=3,
            model_name="llama3.2",
            context_size=4096,
            context_index=1,
            benchmark_type=BenchmarkType.CONTEXT_SCALING,
            benchmark_type_index=2,
            scenario_id="context-scaling-quarter-v1",
            scenario_index=1,
            scenario_version="v1",
            state=RunState.COMPLETED,
            elapsed_ms=12.0,
            metrics={
                "requested_prep_behavior": "calibrated_context_fill",
                "actual_prep_method": "calibration",
                "prep_enforcement_succeeded": True,
                "calibration_status": "exact",
                "eligible_for_strict_aggregate": True,
                "eligible_for_calibrated_context_aggregate": True,
                "eligible_for_ttft_aggregate": True,
                "ttft_ms": 120.0,
                "load_duration_ms": 12.0,
                "generation_tokens_per_second": 24.0,
                "tokens_per_second": 24.0,
            },
        ),
    ]

    summary = build_report_summary(plan={"model_name": "llama3.2"}, environment={}, runs=runs)

    cold_summary = next(
        row for row in summary.benchmark_summaries if row["scenario_id"] == "cold-warm-cold-start-v1"
    )
    context_summary = next(
        row for row in summary.benchmark_summaries if row["scenario_id"] == "context-scaling-quarter-v1"
    )

    assert cold_summary["sample_size"] == 1
    assert cold_summary["strict_sample_size"] == 0
    assert cold_summary["elapsed_ms_sample_size"] == 0
    assert context_summary["sample_size"] == 2
    assert context_summary["strict_sample_size"] == 1
    assert context_summary["elapsed_ms_median"] == 12.0
    assert context_summary["ttft_ms_median"] == 120.0


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


def test_render_markdown_report_surfaces_completed_sample_stats_without_alias_leakage() -> None:
    summary = build_report_summary(
        plan={"model_name": "llama3.2"},
        environment={},
        runs=[
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
                elapsed_ms=10.0,
                metrics={
                    "eligible_for_strict_aggregate": True,
                    "eligible_for_ttft_aggregate": True,
                    "load_duration_ms": 5.0,
                    "ttft_ms": 100.0,
                    "stream_emission_count": 3,
                    "stream_duration_ms": 80.0,
                    "stream_emission_interval_ms_median": 30.0,
                    "stream_emission_interval_ms_p95": 50.0,
                    "stream_output_units_per_second": 37.5,
                    "peak_gpu_memory_mb": 3072.0,
                    "avg_gpu_util_percent": 45.0,
                    "peak_gpu_util_percent": 70.0,
                    "prompt_tokens_per_second": 1000.0,
                    "generation_tokens_per_second": 10.0,
                    "tokens_per_second": 10.0,
                },
            ),
            RunResult(
                run_id="run-0002",
                run_index=2,
                model_name="llama3.2",
                context_size=4096,
                context_index=1,
                benchmark_type="smoke",
                benchmark_type_index=1,
                scenario_id="smoke-basic-v1",
                scenario_index=1,
                state=RunState.COMPLETED,
                elapsed_ms=20.0,
                metrics={
                    "eligible_for_strict_aggregate": True,
                    "eligible_for_ttft_aggregate": True,
                    "load_duration_ms": 7.0,
                    "ttft_ms": 200.0,
                    "stream_emission_count": 5,
                    "stream_duration_ms": 120.0,
                    "stream_emission_interval_ms_median": 25.0,
                    "stream_emission_interval_ms_p95": 35.0,
                    "stream_output_units_per_second": 41.667,
                    "peak_gpu_memory_mb": 4096.0,
                    "avg_gpu_util_percent": 55.0,
                    "peak_gpu_util_percent": 80.0,
                    "prompt_tokens_per_second": 2000.0,
                    "generation_tokens_per_second": 20.0,
                    "tokens_per_second": 20.0,
                },
            ),
        ],
    )

    markdown = render_markdown_report(summary)

    assert "2 of 2 runs completed successfully for llama3.2." in markdown
    assert "By benchmark family:" in markdown
    assert "- Smoke: elapsed median 15.0; prompt tokens/s median 1500.0; generation tokens/s median 15.0." in markdown
    assert "Completed samples: 2 of 2 runs" not in markdown
    assert "Run elapsed time: median 15.0 | p95 20.0 | n=2" in markdown
    assert "TTFT: median 150.0 | p95 200.0 | n=2" in markdown
    assert "Prompt tokens/s: median 1500.0 | p95 2000.0 | n=2" in markdown
    assert "Generation tokens/s: median 15.0 | p95 20.0 | n=2" in markdown
    assert "Load duration: median 6.0 | p95 7.0 | n=2" in markdown
    assert "stream_emission_interval_ms_median: median 27.5 | p95 30.0 | n=2" not in markdown
    assert "Median stream emission interval: median 27.5 | p95 30.0 | n=2" not in markdown
    assert "Peak GPU memory: median 3584.0 | p95 4096.0 | n=2" not in markdown
    assert "\n- tokens_per_second:" not in markdown
    assert "| tokens_per_second |" not in markdown


def test_render_markdown_report_summarizes_benchmark_families_separately() -> None:
    summary = build_report_summary(
        plan={"model_name": "llama3.2"},
        environment={},
        runs=[
            RunResult(
                run_id="run-smoke",
                run_index=1,
                model_name="llama3.2",
                context_size=4096,
                context_index=1,
                benchmark_type=BenchmarkType.SMOKE,
                benchmark_type_index=1,
                scenario_id="smoke-basic-v1",
                scenario_index=1,
                state=RunState.COMPLETED,
                elapsed_ms=10.0,
                metrics={
                    "eligible_for_strict_aggregate": True,
                    "prompt_tokens_per_second": 1000.0,
                    "generation_tokens_per_second": 20.0,
                },
            ),
            RunResult(
                run_id="run-ttft",
                run_index=2,
                model_name="llama3.2",
                context_size=4096,
                context_index=1,
                benchmark_type=BenchmarkType.TTFT,
                benchmark_type_index=2,
                scenario_id="ttft-short-v1",
                scenario_index=1,
                state=RunState.COMPLETED,
                elapsed_ms=30.0,
                metrics={
                    "eligible_for_strict_aggregate": True,
                    "eligible_for_ttft_aggregate": True,
                    "ttft_ms": 80.0,
                },
            ),
            RunResult(
                run_id="run-concurrency-p2",
                run_index=3,
                model_name="llama3.2",
                context_size=4096,
                context_index=1,
                benchmark_type=BenchmarkType.CONCURRENCY_SMOKE,
                benchmark_type_index=3,
                scenario_id="concurrency-smoke-p2-v1",
                scenario_index=1,
                state=RunState.COMPLETED,
                elapsed_ms=160.0,
                metrics={
                    "eligible_for_strict_aggregate": True,
                    "concurrency_parallelism": 2,
                    "concurrency_request_ttft_ms_p50": 20.0,
                },
            ),
            RunResult(
                run_id="run-concurrency-p4",
                run_index=4,
                model_name="llama3.2",
                context_size=4096,
                context_index=1,
                benchmark_type=BenchmarkType.CONCURRENCY_SMOKE,
                benchmark_type_index=3,
                scenario_id="concurrency-smoke-p4-v1",
                scenario_index=2,
                state=RunState.COMPLETED,
                elapsed_ms=260.0,
                metrics={
                    "eligible_for_strict_aggregate": True,
                    "concurrency_parallelism": 4,
                    "concurrency_request_ttft_ms_p50": 35.0,
                },
            ),
        ],
    )

    markdown = render_markdown_report(summary)
    executive_section = markdown.split("## Per-model summary", maxsplit=1)[0]

    assert "4 of 4 runs completed successfully for llama3.2." in executive_section
    assert "By benchmark family:" in executive_section
    assert "- Smoke: elapsed median 10.0; prompt tokens/s median 1000.0; generation tokens/s median 20.0." in executive_section
    assert "- TTFT: TTFT median 80.0; elapsed median 30.0." in executive_section
    assert "- Concurrency-smoke: p=2 elapsed median 160.0; p=4 elapsed median 260.0; request TTFT p50 median 27.5." in executive_section
    assert "- Run elapsed time: median 95.0" not in executive_section
    assert "- TTFT: median 80.0" not in executive_section


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
