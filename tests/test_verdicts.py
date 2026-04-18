from __future__ import annotations

from ollama_workload_profiler.models.plan import BenchmarkType
from ollama_workload_profiler.models.results import RunResult, RunState
from ollama_workload_profiler.models.verdicts import VerdictLabel
from ollama_workload_profiler.reporting.summary import build_report_summary
from ollama_workload_profiler.reporting.verdicts import classify_verdict


def test_failure_maps_to_avoid_on_this_hardware() -> None:
    verdict = classify_verdict(
        success=False,
        ttft_ms=9000,
        gen_tokens_per_second=1.0,
        ram_headroom_gb=0.2,
    )

    assert verdict.label is VerdictLabel.AVOID_ON_THIS_HARDWARE


def test_missing_metrics_do_not_become_optimistic() -> None:
    verdict = classify_verdict(success=True)

    assert verdict.label is VerdictLabel.USE_WITH_CAUTION


def test_verdict_display_labels_match_spec() -> None:
    assert [label.display_label for label in VerdictLabel] == [
        "Good fit",
        "Good fit with caution",
        "Use with caution",
        "Use only for short tasks",
        "Avoid on this hardware",
    ]


def test_use_case_suitability_matrix_is_structured_and_marks_missing_contexts() -> None:
    runs = [
        RunResult(
            run_id="run-0001",
            run_index=1,
            model_name="llama3.2",
            context_size=4096,
            context_index=1,
            benchmark_type=BenchmarkType.USE_CASE_PROFILES,
            benchmark_type_index=1,
            scenario_id="use-case-profiles-quick-qa-v1",
            scenario_index=1,
            scenario_version="v1",
            scenario_name="Quick QA",
            state=RunState.COMPLETED,
            elapsed_ms=910.0,
            metrics={"tokens_per_second": 22.0},
        ),
        RunResult(
            run_id="run-0002",
            run_index=2,
            model_name="llama3.2",
            context_size=4096,
            context_index=1,
            benchmark_type=BenchmarkType.USE_CASE_PROFILES,
            benchmark_type_index=1,
            scenario_id="use-case-profiles-quick-qa-v1",
            scenario_index=1,
            scenario_version="v1",
            scenario_name="Quick QA",
            state=RunState.COMPLETED,
            elapsed_ms=995.0,
            metrics={"tokens_per_second": 18.0},
        ),
        RunResult(
            run_id="run-0003",
            run_index=3,
            model_name="llama3.2",
            context_size=4096,
            context_index=1,
            benchmark_type=BenchmarkType.USE_CASE_PROFILES,
            benchmark_type_index=1,
            scenario_id="use-case-profiles-quick-qa-v1",
            scenario_index=1,
            scenario_version="v1",
            scenario_name="Quick QA",
            state=RunState.FAILED,
            metrics={"finalization_error": "sampler shutdown failed"},
        ),
        RunResult(
            run_id="run-0004",
            run_index=4,
            model_name="llama3.2",
            context_size=8192,
            context_index=2,
            benchmark_type=BenchmarkType.USE_CASE_PROFILES,
            benchmark_type_index=1,
            scenario_id="use-case-profiles-quick-qa-v1",
            scenario_index=1,
            scenario_version="v1",
            scenario_name="Quick QA",
            state=RunState.COMPLETED,
            elapsed_ms=1430.0,
            metrics={},
        ),
        RunResult(
            run_id="run-0006",
            run_index=6,
            model_name="llama3.2",
            context_size=16384,
            context_index=3,
            benchmark_type=BenchmarkType.SMOKE,
            benchmark_type_index=2,
            scenario_id="smoke-basic-v1",
            scenario_index=1,
            scenario_version="v1",
            scenario_name="Smoke",
            state=RunState.COMPLETED,
            elapsed_ms=20.0,
            metrics={"tokens_per_second": 120.0},
        ),
        RunResult(
            run_id="run-0005",
            run_index=5,
            model_name="llama3.2",
            context_size=4096,
            context_index=1,
            benchmark_type=BenchmarkType.SMOKE,
            benchmark_type_index=2,
            scenario_id="smoke-basic-v1",
            scenario_index=1,
            scenario_version="v1",
            scenario_name="Smoke",
            state=RunState.COMPLETED,
            elapsed_ms=25.0,
            metrics={"tokens_per_second": 100.0},
        ),
    ]

    summary = build_report_summary(
        plan={
            "model_name": "llama3.2",
            "contexts": [4096, 8192, 16384],
            "benchmark_types": [BenchmarkType.USE_CASE_PROFILES, BenchmarkType.SMOKE],
        },
        environment={"platform": "test"},
        runs=runs,
    )

    assert len(summary.use_case_matrix) == 6

    row = summary.use_case_matrix[0]
    assert row["use_case_profile"] == "Quick QA"
    assert row["scenario_id"] == "use-case-profiles-quick-qa-v1"
    assert list(row) == [
        "use_case_profile",
        "scenario_id",
        "4096",
        "8192",
        "16384",
    ]
    cell_4096 = row["4096"]
    assert cell_4096.verdict_label is VerdictLabel.GOOD_FIT_WITH_CAUTION
    assert cell_4096.supporting_tuple == (
        3,
        3,
        2,
        1,
        0,
        952.5,
        20.0,
        None,
        None,
        0.333,
    )

    cell_8192 = row["8192"]
    assert cell_8192.verdict_label is VerdictLabel.USE_WITH_CAUTION
    assert cell_8192.supporting_tuple == (
        1,
        1,
        1,
        0,
        0,
        1430.0,
        None,
        None,
        None,
        0.0,
    )

    cell_16384 = row["16384"]
    assert cell_16384.verdict_label is None
    assert cell_16384.missing_planned_context is True
    assert cell_16384.supporting_tuple == (0, 0, 0, 0, 0, None, None, None, None, None)

    missing_row = summary.use_case_matrix[1]
    assert missing_row["use_case_profile"] == "Long document summary"
    assert missing_row["4096"].verdict_label is None
    assert missing_row["4096"].missing_planned_context is True
