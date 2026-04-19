from __future__ import annotations

import json
from pathlib import Path

import pytest

from ollama_workload_profiler.methodology import BENCHMARK_METHODOLOGY_VERSION
from ollama_workload_profiler.reporting.compare import (
    CompareArtifactError,
    CompareGateStatus,
    CompareWarning,
    MetricDelta,
    MissingMetricReason,
    RowKey,
    WarningSeverity,
    WarningScope,
    assess_execution_policy_compatibility,
    assess_environment_runtime_comparability,
    assess_methodology_compatibility,
    build_row_metric_deltas,
    build_session_metric_deltas,
    compare_sessions,
    match_benchmark_rows,
    load_compare_session,
    render_compare_json,
    render_compare_text,
    strict_mode_failure_reasons,
)


def test_compare_warning_has_stable_machine_readable_fields() -> None:
    warning = CompareWarning(
        code="execution_policy.temperature_mismatch",
        severity=WarningSeverity.MAJOR,
        message="Execution setting temperature differs.",
        scope=WarningScope.EXECUTION_POLICY,
        field="execution_settings.temperature",
        baseline=0.0,
        candidate=0.2,
    )

    assert warning.model_dump(mode="json") == {
        "code": "execution_policy.temperature_mismatch",
        "severity": "major",
        "message": "Execution setting temperature differs.",
        "scope": "execution_policy",
        "field": "execution_settings.temperature",
        "baseline": 0.0,
        "candidate": 0.2,
    }


def test_row_key_serializes_to_stable_comparison_identity() -> None:
    row_key = RowKey(
        model_name="llama3.2",
        context_size=4096,
        benchmark_type="smoke",
        scenario_id="smoke-basic-v1",
    )

    assert row_key.as_sort_tuple() == ("llama3.2", 4096, "smoke", "smoke-basic-v1")
    assert row_key.display() == "llama3.2 | 4096 | smoke | smoke-basic-v1"


def test_strict_mode_failure_mapping_is_explicit() -> None:
    warnings = [
        CompareWarning(
            code="methodology.contract_unavailable",
            severity=WarningSeverity.MAJOR,
            message="Sessions are not directly comparable under the same benchmark contract.",
            scope=WarningScope.METHODOLOGY,
        ),
        CompareWarning(
            code="environment.hostname_mismatch",
            severity=WarningSeverity.MINOR,
            message="Hostname differs.",
            scope=WarningScope.ENVIRONMENT,
        ),
    ]

    assert strict_mode_failure_reasons(
        warnings=warnings,
        gates={
            "methodology_compatibility": CompareGateStatus.WARN,
            "execution_policy_compatibility": CompareGateStatus.PASS,
            "artifact_schema_compatibility": CompareGateStatus.PASS,
            "environment_runtime_comparability": CompareGateStatus.WARN,
        },
        unmatched_row_count=0,
    ) == ["methodology.contract_unavailable"]


def test_missing_metric_reasons_are_stable() -> None:
    assert [reason.value for reason in MissingMetricReason] == [
        "available",
        "missing",
        "not_applicable",
        "ineligible",
        "schema_unavailable",
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_minimal_session(path: Path) -> None:
    path.mkdir()
    _write_json(
        path / "summary.json",
        {
            "benchmark_summaries": [
                {
                    "model_name": "llama3.2",
                    "context_size": 4096,
                    "benchmark_type": "smoke",
                    "scenario_id": "smoke-basic-v1",
                    "elapsed_ms_median": 100.0,
                }
            ]
        },
    )
    _write_json(
        path / "plan.json",
        {
            "model_name": "llama3.2",
            "contexts": [4096],
            "benchmark_types": ["smoke"],
            "execution_settings": {
                "seed": 42,
                "temperature": 0.0,
                "top_p": None,
                "repetitions": 1,
                "warmup_runs": 1,
                "warmup_enabled": True,
            },
        },
    )
    _write_json(
        path / "environment.json",
        {
            "execution_settings": {
                "seed": 42,
                "temperature": 0.0,
                "top_p": None,
                "repetitions": 1,
                "warmup_runs": 1,
                "warmup_enabled": True,
            }
        },
    )


def _write_upgraded_session(path: Path) -> None:
    path.mkdir()
    execution_settings = {
        "seed": 42,
        "temperature": 0.0,
        "top_p": None,
        "repetitions": 1,
        "warmup_runs": 1,
        "warmup_enabled": True,
    }
    _write_json(
        path / "summary.json",
        {
            "benchmark_methodology_version": BENCHMARK_METHODOLOGY_VERSION,
            "benchmark_summaries": [
                {
                    "model_name": "llama3.2",
                    "context_size": 4096,
                    "benchmark_type": "smoke",
                    "scenario_id": "smoke-basic-v1",
                    "strict_sample_size": 1,
                    "sample_size": 1,
                    "completed_runs": 1,
                    "failed_runs": 0,
                    "elapsed_ms_median": 100.0,
                    "elapsed_ms_p95": 100.0,
                    "elapsed_ms_sample_size": 1,
                    "generation_tokens_per_second_median": 20.0,
                    "generation_tokens_per_second_p95": 20.0,
                    "generation_tokens_per_second_sample_size": 1,
                }
            ],
            "session_metrics": {
                "run_count": 1,
                "completed_runs": 1,
                "failed_runs": 0,
                "stopped_runs": 0,
                "completed_sample_size": 1,
                "elapsed_ms_median": 100.0,
                "elapsed_ms_p95": 100.0,
                "elapsed_ms_sample_size": 1,
                "generation_tokens_per_second_median": 20.0,
                "generation_tokens_per_second_p95": 20.0,
                "generation_tokens_per_second_sample_size": 1,
            },
        },
    )
    _write_json(
        path / "plan.json",
        {
            "benchmark_methodology_version": BENCHMARK_METHODOLOGY_VERSION,
            "model_name": "llama3.2",
            "contexts": [4096],
            "benchmark_types": ["smoke"],
            "execution_settings": execution_settings,
        },
    )
    _write_json(
        path / "environment.json",
        {
            "benchmark_methodology_version": BENCHMARK_METHODOLOGY_VERSION,
            "execution_settings": execution_settings,
        },
    )


def test_load_compare_session_reads_required_artifacts(tmp_path: Path) -> None:
    session_dir = tmp_path / "session-a"
    _write_minimal_session(session_dir)

    loaded = load_compare_session(session_dir)

    assert loaded.path == session_dir
    assert loaded.summary["benchmark_summaries"][0]["scenario_id"] == "smoke-basic-v1"
    assert loaded.plan["model_name"] == "llama3.2"
    assert loaded.environment["execution_settings"]["seed"] == 42


def test_load_compare_session_rejects_missing_required_artifact(tmp_path: Path) -> None:
    session_dir = tmp_path / "session-a"
    _write_minimal_session(session_dir)
    (session_dir / "environment.json").unlink()

    with pytest.raises(CompareArtifactError, match="Missing required artifact: environment.json"):
        load_compare_session(session_dir)


def test_load_compare_session_rejects_malformed_json(tmp_path: Path) -> None:
    session_dir = tmp_path / "session-a"
    _write_minimal_session(session_dir)
    (session_dir / "summary.json").write_text("{", encoding="utf-8")

    with pytest.raises(CompareArtifactError, match="Malformed JSON artifact: summary.json"):
        load_compare_session(session_dir)


def test_load_compare_session_rejects_invalid_summary_shape(tmp_path: Path) -> None:
    session_dir = tmp_path / "session-a"
    _write_minimal_session(session_dir)
    _write_json(session_dir / "summary.json", {"benchmark_summaries": {}})

    with pytest.raises(CompareArtifactError, match="summary.json must contain benchmark_summaries as a list"):
        load_compare_session(session_dir)


def test_methodology_artifact_consistency_passes_when_all_versions_are_equal(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session-a"
    _write_upgraded_session(session_dir)

    loaded = load_compare_session(session_dir)

    assert loaded.methodology_version == BENCHMARK_METHODOLOGY_VERSION
    assert loaded.methodology_warnings == []


def test_methodology_artifact_consistency_warns_when_one_version_is_missing(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session-a"
    _write_upgraded_session(session_dir)
    summary_payload = json.loads((session_dir / "summary.json").read_text(encoding="utf-8"))
    summary_payload.pop("benchmark_methodology_version")
    _write_json(session_dir / "summary.json", summary_payload)

    loaded = load_compare_session(session_dir)

    assert loaded.methodology_version == BENCHMARK_METHODOLOGY_VERSION
    assert [warning.code for warning in loaded.methodology_warnings] == [
        "methodology.artifact_version_missing"
    ]
    assert loaded.methodology_warnings[0].severity is WarningSeverity.MINOR
    assert loaded.methodology_warnings[0].field == "summary.json.benchmark_methodology_version"


def test_methodology_artifact_consistency_warns_when_versions_conflict(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session-a"
    _write_upgraded_session(session_dir)
    environment_payload = json.loads((session_dir / "environment.json").read_text(encoding="utf-8"))
    environment_payload["benchmark_methodology_version"] = "bmk-v3"
    _write_json(session_dir / "environment.json", environment_payload)

    loaded = load_compare_session(session_dir)

    assert loaded.methodology_version == BENCHMARK_METHODOLOGY_VERSION
    assert [warning.code for warning in loaded.methodology_warnings] == [
        "methodology.artifact_version_conflict"
    ]
    assert loaded.methodology_warnings[0].severity is WarningSeverity.MAJOR
    assert loaded.methodology_warnings[0].field == "benchmark_methodology_version"
    assert loaded.methodology_warnings[0].baseline == {
        "environment.json": "bmk-v3",
        "plan.json": BENCHMARK_METHODOLOGY_VERSION,
        "summary.json": BENCHMARK_METHODOLOGY_VERSION,
    }


def test_methodology_compatibility_passes_for_upgraded_contract_sessions(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_upgraded_session(baseline_dir)
    _write_upgraded_session(candidate_dir)

    gate, warnings = assess_methodology_compatibility(
        baseline=load_compare_session(baseline_dir),
        candidate=load_compare_session(candidate_dir),
    )

    assert gate is CompareGateStatus.PASS
    assert warnings == []


def test_methodology_compatibility_uses_matching_explicit_versions_as_strong_signal(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_minimal_session(baseline_dir)
    _write_minimal_session(candidate_dir)
    for session_dir in (baseline_dir, candidate_dir):
        for artifact_name in ("summary.json", "plan.json", "environment.json"):
            payload = json.loads((session_dir / artifact_name).read_text(encoding="utf-8"))
            payload["benchmark_methodology_version"] = BENCHMARK_METHODOLOGY_VERSION
            _write_json(session_dir / artifact_name, payload)

    gate, warnings = assess_methodology_compatibility(
        baseline=load_compare_session(baseline_dir),
        candidate=load_compare_session(candidate_dir),
    )

    assert gate is CompareGateStatus.PASS
    assert warnings == []


def test_compare_sessions_includes_intra_session_methodology_warnings(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_upgraded_session(baseline_dir)
    _write_upgraded_session(candidate_dir)
    candidate_environment = json.loads((candidate_dir / "environment.json").read_text(encoding="utf-8"))
    candidate_environment["benchmark_methodology_version"] = "bmk-v3"
    _write_json(candidate_dir / "environment.json", candidate_environment)

    result = compare_sessions(baseline_dir=baseline_dir, candidate_dir=candidate_dir)

    assert result.comparability_status is CompareGateStatus.WARN
    assert "methodology.artifact_version_conflict" in [warning.code for warning in result.warnings]
    assert "methodology.artifact_version_conflict" in result.strict_failure_reasons


def test_methodology_compatibility_warns_for_different_explicit_versions(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_upgraded_session(baseline_dir)
    _write_upgraded_session(candidate_dir)
    candidate_plan = json.loads((candidate_dir / "plan.json").read_text(encoding="utf-8"))
    candidate_plan["benchmark_methodology_version"] = "bmk-v3"
    _write_json(candidate_dir / "plan.json", candidate_plan)

    gate, warnings = assess_methodology_compatibility(
        baseline=load_compare_session(baseline_dir),
        candidate=load_compare_session(candidate_dir),
    )

    assert gate is CompareGateStatus.WARN
    assert [warning.code for warning in warnings] == ["methodology.version_mismatch"]
    assert warnings[0].severity is WarningSeverity.MAJOR
    assert warnings[0].field == "benchmark_methodology_version"
    assert warnings[0].baseline == BENCHMARK_METHODOLOGY_VERSION
    assert warnings[0].candidate == "bmk-v3"


def test_methodology_compatibility_warns_for_pre_v020_like_sessions(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_minimal_session(baseline_dir)
    _write_upgraded_session(candidate_dir)

    gate, warnings = assess_methodology_compatibility(
        baseline=load_compare_session(baseline_dir),
        candidate=load_compare_session(candidate_dir),
    )

    assert gate is CompareGateStatus.WARN
    assert [warning.code for warning in warnings] == ["methodology.contract_unavailable"]
    assert warnings[0].severity is WarningSeverity.MAJOR
    assert warnings[0].scope is WarningScope.METHODOLOGY
    assert "not directly comparable under the same benchmark contract" in warnings[0].message


def test_match_benchmark_rows_groups_by_stable_row_key(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_upgraded_session(baseline_dir)
    _write_upgraded_session(candidate_dir)

    _write_json(
        baseline_dir / "summary.json",
        {
            "benchmark_summaries": [
                {
                    "model_name": "llama3.2",
                    "context_size": 8192,
                    "benchmark_type": "ttft",
                    "scenario_id": "ttft-basic-v1",
                    "strict_sample_size": 1,
                },
                {
                    "model_name": "llama3.2",
                    "context_size": 4096,
                    "benchmark_type": "smoke",
                    "scenario_id": "smoke-basic-v1",
                    "strict_sample_size": 1,
                },
            ]
        },
    )
    _write_json(
        candidate_dir / "summary.json",
        {
            "benchmark_summaries": [
                {
                    "model_name": "llama3.2",
                    "context_size": 4096,
                    "benchmark_type": "smoke",
                    "scenario_id": "smoke-basic-v1",
                    "strict_sample_size": 1,
                },
                {
                    "model_name": "llama3.2",
                    "context_size": 4096,
                    "benchmark_type": "prompt-scaling",
                    "scenario_id": "prompt-scaling-small-v1",
                    "strict_sample_size": 1,
                },
            ]
        },
    )

    result = match_benchmark_rows(
        baseline=load_compare_session(baseline_dir),
        candidate=load_compare_session(candidate_dir),
    )

    assert [match.key.display() for match in result.matched] == [
        "llama3.2 | 4096 | smoke | smoke-basic-v1"
    ]
    assert [key.display() for key in result.only_in_baseline] == [
        "llama3.2 | 8192 | ttft | ttft-basic-v1"
    ]
    assert [key.display() for key in result.only_in_candidate] == [
        "llama3.2 | 4096 | prompt-scaling | prompt-scaling-small-v1"
    ]
    assert [warning.code for warning in result.warnings] == ["row_coverage.unmatched_rows"]
    assert result.warnings[0].severity is WarningSeverity.MAJOR


def test_match_benchmark_rows_rejects_duplicate_keys(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_upgraded_session(baseline_dir)
    _write_upgraded_session(candidate_dir)
    duplicate_row = {
        "model_name": "llama3.2",
        "context_size": 4096,
        "benchmark_type": "smoke",
        "scenario_id": "smoke-basic-v1",
        "strict_sample_size": 1,
    }
    _write_json(
        baseline_dir / "summary.json",
        {"benchmark_summaries": [duplicate_row, duplicate_row]},
    )

    with pytest.raises(CompareArtifactError, match="Duplicate benchmark summary row key"):
        match_benchmark_rows(
            baseline=load_compare_session(baseline_dir),
            candidate=load_compare_session(candidate_dir),
        )


def test_execution_policy_compatibility_passes_for_matching_settings(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_upgraded_session(baseline_dir)
    _write_upgraded_session(candidate_dir)

    gate, warnings = assess_execution_policy_compatibility(
        baseline=load_compare_session(baseline_dir),
        candidate=load_compare_session(candidate_dir),
    )

    assert gate is CompareGateStatus.PASS
    assert warnings == []


def test_execution_policy_compatibility_warns_for_setting_mismatches(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_upgraded_session(baseline_dir)
    _write_upgraded_session(candidate_dir)
    candidate_plan = json.loads((candidate_dir / "plan.json").read_text(encoding="utf-8"))
    candidate_plan["execution_settings"]["temperature"] = 0.2
    candidate_plan["execution_settings"]["warmup_enabled"] = False
    _write_json(candidate_dir / "plan.json", candidate_plan)

    gate, warnings = assess_execution_policy_compatibility(
        baseline=load_compare_session(baseline_dir),
        candidate=load_compare_session(candidate_dir),
    )

    assert gate is CompareGateStatus.WARN
    assert [warning.code for warning in warnings] == [
        "execution_policy.temperature_mismatch",
        "execution_policy.warmup_enabled_mismatch",
    ]
    assert all(warning.severity is WarningSeverity.MAJOR for warning in warnings)
    assert all(warning.scope is WarningScope.EXECUTION_POLICY for warning in warnings)
    assert warnings[0].field == "execution_settings.temperature"
    assert warnings[0].baseline == 0.0
    assert warnings[0].candidate == 0.2


def test_execution_policy_compatibility_warns_when_settings_are_unavailable(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_upgraded_session(baseline_dir)
    _write_upgraded_session(candidate_dir)
    baseline_plan = json.loads((baseline_dir / "plan.json").read_text(encoding="utf-8"))
    baseline_plan.pop("execution_settings")
    _write_json(baseline_dir / "plan.json", baseline_plan)

    gate, warnings = assess_execution_policy_compatibility(
        baseline=load_compare_session(baseline_dir),
        candidate=load_compare_session(candidate_dir),
    )

    assert gate is CompareGateStatus.WARN
    assert [warning.code for warning in warnings] == ["execution_policy.execution_settings_unavailable"]
    assert warnings[0].severity is WarningSeverity.MAJOR


def test_environment_runtime_comparability_classifies_major_runtime_mismatches(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_upgraded_session(baseline_dir)
    _write_upgraded_session(candidate_dir)
    _write_json(
        baseline_dir / "environment.json",
        {
            "execution_settings": json.loads((baseline_dir / "plan.json").read_text(encoding="utf-8"))[
                "execution_settings"
            ],
            "ollama": {
                "version": {"value": "0.7.0"},
                "selected_model": {
                    "details": {"quantization_level": "Q4_K_M"},
                    "model_info": {"context_length": 8192},
                },
            },
            "accelerator": {"vendor": "nvidia", "name": "RTX 4090", "vram_mb": 24564},
        },
    )
    _write_json(
        candidate_dir / "environment.json",
        {
            "execution_settings": json.loads((candidate_dir / "plan.json").read_text(encoding="utf-8"))[
                "execution_settings"
            ],
            "ollama": {
                "version": {"value": "0.7.1"},
                "selected_model": {
                    "details": {"quantization_level": "Q8_0"},
                    "model_info": {"context_length": 8192},
                },
            },
            "accelerator": {"vendor": "nvidia", "name": "RTX 4080", "vram_mb": 16384},
        },
    )

    gate, warnings = assess_environment_runtime_comparability(
        baseline=load_compare_session(baseline_dir),
        candidate=load_compare_session(candidate_dir),
    )

    assert gate is CompareGateStatus.WARN
    assert [warning.code for warning in warnings] == [
        "environment.ollama_version_mismatch",
        "environment.model_details_mismatch",
        "environment.accelerator_mismatch",
    ]
    assert all(warning.severity is WarningSeverity.MAJOR for warning in warnings)
    assert all(warning.scope is WarningScope.ENVIRONMENT for warning in warnings)


def test_environment_runtime_comparability_classifies_minor_host_mismatches(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_upgraded_session(baseline_dir)
    _write_upgraded_session(candidate_dir)
    baseline_environment = json.loads((baseline_dir / "environment.json").read_text(encoding="utf-8"))
    candidate_environment = json.loads((candidate_dir / "environment.json").read_text(encoding="utf-8"))
    baseline_environment.update({"host": {"hostname": "bench-a"}, "python_version": "3.13.1"})
    candidate_environment.update({"host": {"hostname": "bench-b"}, "python_version": "3.13.2"})
    _write_json(baseline_dir / "environment.json", baseline_environment)
    _write_json(candidate_dir / "environment.json", candidate_environment)

    gate, warnings = assess_environment_runtime_comparability(
        baseline=load_compare_session(baseline_dir),
        candidate=load_compare_session(candidate_dir),
    )

    assert gate is CompareGateStatus.WARN
    assert [warning.code for warning in warnings] == [
        "environment.hostname_mismatch",
        "environment.python_version_mismatch",
    ]
    assert all(warning.severity is WarningSeverity.MINOR for warning in warnings)


def test_row_metric_deltas_use_deterministic_order_and_numeric_delta() -> None:
    baseline = {
        "elapsed_ms_median": 100.0,
        "elapsed_ms_p95": 150.0,
        "generation_tokens_per_second_median": 20.0,
        "generation_tokens_per_second_p95": 25.0,
        "generation_tokens_per_second_sample_size": 2,
    }
    candidate = {
        "elapsed_ms_median": 80.0,
        "elapsed_ms_p95": 120.0,
        "generation_tokens_per_second_median": 22.0,
        "generation_tokens_per_second_p95": 28.0,
        "generation_tokens_per_second_sample_size": 2,
    }

    deltas = build_row_metric_deltas(baseline=baseline, candidate=candidate)

    assert [delta.metric for delta in deltas[:4]] == [
        "elapsed_ms_median",
        "elapsed_ms_p95",
        "prompt_tokens_per_second_median",
        "prompt_tokens_per_second_p95",
    ]
    elapsed = deltas[0]
    assert elapsed == MetricDelta(
        scope="row",
        metric="elapsed_ms_median",
        baseline=100.0,
        candidate=80.0,
        delta=-20.0,
        delta_percent=-20.0,
        reason=MissingMetricReason.AVAILABLE,
    )
    generation = next(delta for delta in deltas if delta.metric == "generation_tokens_per_second_median")
    assert generation.delta == 2.0
    assert generation.delta_percent == 10.0


def test_metric_deltas_preserve_missing_reasons() -> None:
    baseline = {
        "elapsed_ms_median": 0.0,
        "elapsed_ms_sample_size": 1,
        "prompt_tokens_per_second_sample_size": 0,
        "strict_sample_size": 0,
    }
    candidate = {
        "elapsed_ms_median": 10.0,
        "elapsed_ms_sample_size": 1,
        "prompt_tokens_per_second_median": 100.0,
        "prompt_tokens_per_second_sample_size": 1,
        "strict_sample_size": 1,
    }

    deltas = build_row_metric_deltas(baseline=baseline, candidate=candidate)

    elapsed = next(delta for delta in deltas if delta.metric == "elapsed_ms_median")
    assert elapsed.reason is MissingMetricReason.AVAILABLE
    assert elapsed.delta == 10.0
    assert elapsed.delta_percent is None

    prompt = next(delta for delta in deltas if delta.metric == "prompt_tokens_per_second_median")
    assert prompt.reason is MissingMetricReason.INELIGIBLE
    assert prompt.baseline is None
    assert prompt.candidate == 100.0
    assert prompt.delta is None

    ttft = next(delta for delta in deltas if delta.metric == "ttft_ms_median")
    assert ttft.reason is MissingMetricReason.NOT_APPLICABLE


def test_session_metric_deltas_render_unavailable_when_either_side_lacks_field() -> None:
    baseline = {
        "completed_runs": 3,
        "failed_runs": 0,
        "elapsed_ms_median": 100.0,
        "elapsed_ms_sample_size": 3,
    }
    candidate = {
        "completed_runs": 3,
        "failed_runs": 1,
        "elapsed_ms_sample_size": 3,
    }

    deltas = build_session_metric_deltas(baseline=baseline, candidate=candidate)

    assert [delta.metric for delta in deltas[:4]] == [
        "run_count",
        "completed_runs",
        "failed_runs",
        "stopped_runs",
    ]
    completed = next(delta for delta in deltas if delta.metric == "completed_runs")
    assert completed.reason is MissingMetricReason.AVAILABLE
    assert completed.delta == 0.0

    elapsed = next(delta for delta in deltas if delta.metric == "elapsed_ms_median")
    assert elapsed.reason is MissingMetricReason.MISSING
    assert elapsed.baseline == 100.0
    assert elapsed.candidate is None


def test_compare_sessions_builds_structured_result_with_gates_and_deltas(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_upgraded_session(baseline_dir)
    _write_upgraded_session(candidate_dir)
    candidate_summary = json.loads((candidate_dir / "summary.json").read_text(encoding="utf-8"))
    candidate_summary["session_metrics"]["elapsed_ms_median"] = 80.0
    candidate_summary["benchmark_summaries"][0]["elapsed_ms_median"] = 80.0
    _write_json(candidate_dir / "summary.json", candidate_summary)

    result = compare_sessions(baseline_dir=baseline_dir, candidate_dir=candidate_dir)

    assert result.baseline_path == str(baseline_dir)
    assert result.candidate_path == str(candidate_dir)
    assert result.comparability_status is CompareGateStatus.PASS
    assert result.gates == {
        "methodology_compatibility": CompareGateStatus.PASS,
        "execution_policy_compatibility": CompareGateStatus.PASS,
        "artifact_schema_compatibility": CompareGateStatus.PASS,
        "environment_runtime_comparability": CompareGateStatus.PASS,
    }
    assert result.strict_failure_reasons == []
    assert result.row_comparisons[0].key.display() == "llama3.2 | 4096 | smoke | smoke-basic-v1"
    assert result.session_summary_deltas[5].metric == "elapsed_ms_median"
    assert result.session_summary_deltas[5].delta == -20.0


def test_render_compare_text_puts_warnings_before_metric_deltas_and_groups_rows(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_minimal_session(baseline_dir)
    _write_upgraded_session(candidate_dir)

    result = compare_sessions(baseline_dir=baseline_dir, candidate_dir=candidate_dir)
    text = render_compare_text(result)

    assert f"Baseline: {baseline_dir}" in text
    assert f"Candidate: {candidate_dir}" in text
    assert "Comparability: warn" in text
    assert "Methodology: warn" in text
    assert f"Benchmark methodology: baseline unavailable -> candidate {BENCHMARK_METHODOLOGY_VERSION}" in text
    assert "Warnings:" in text
    assert "[major] methodology.contract_unavailable" in text
    assert "Session Summary:" in text
    assert "Rows:" in text
    assert "Rows without changed metrics: 1" in text
    assert "Unmatched Rows:" in text
    assert text.index("Warnings:") < text.index("Session Summary:")
    assert text.index("Session Summary:") < text.index("Rows:")


def test_render_compare_text_defaults_to_changed_metrics_only(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_upgraded_session(baseline_dir)
    _write_upgraded_session(candidate_dir)
    candidate_summary = json.loads((candidate_dir / "summary.json").read_text(encoding="utf-8"))
    candidate_summary["session_metrics"]["elapsed_ms_median"] = 80.0
    candidate_summary["benchmark_summaries"][0]["elapsed_ms_median"] = 80.0
    _write_json(candidate_dir / "summary.json", candidate_summary)

    text = render_compare_text(compare_sessions(baseline_dir=baseline_dir, candidate_dir=candidate_dir))

    assert "Result Summary:" in text
    assert "- Matched rows: 1" in text
    assert "- Warnings: none" in text
    assert "elapsed_ms_median: 100 ms -> 80 ms (delta -20 ms, -20%)" in text
    assert "| elapsed_ms_median | 100 ms | 80 ms | -20 ms | -20% | available |" in text
    assert "prompt_tokens_per_second_median" not in text
    assert "generation_tokens_per_second_median" not in text


def test_render_compare_text_collapses_default_rows_without_changed_metrics(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_upgraded_session(baseline_dir)
    _write_upgraded_session(candidate_dir)

    text = render_compare_text(compare_sessions(baseline_dir=baseline_dir, candidate_dir=candidate_dir))

    assert "Rows without changed metrics: 1" in text
    assert "model=llama3.2 ctx=4096 benchmark=smoke scenario=smoke-basic-v1" not in text
    assert "No changed metrics." in text


def test_render_compare_text_all_metrics_preserves_full_metric_tables(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_upgraded_session(baseline_dir)
    _write_upgraded_session(candidate_dir)

    text = render_compare_text(
        compare_sessions(baseline_dir=baseline_dir, candidate_dir=candidate_dir),
        all_metrics=True,
    )

    assert "| run_count | 1 | 1 | 0 | 0% | available |" in text
    assert "prompt_tokens_per_second_median" in text
    assert "generation_tokens_per_second_median" in text
    assert "No changed metrics." not in text


def test_render_compare_text_warning_lines_include_values_when_present(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_upgraded_session(baseline_dir)
    _write_upgraded_session(candidate_dir)
    baseline_environment = json.loads((baseline_dir / "environment.json").read_text(encoding="utf-8"))
    candidate_environment = json.loads((candidate_dir / "environment.json").read_text(encoding="utf-8"))
    baseline_environment["host"] = {"hostname": "bench-a"}
    candidate_environment["host"] = {"hostname": "bench-b"}
    _write_json(baseline_dir / "environment.json", baseline_environment)
    _write_json(candidate_dir / "environment.json", candidate_environment)

    text = render_compare_text(compare_sessions(baseline_dir=baseline_dir, candidate_dir=candidate_dir))

    assert "[minor] environment.hostname_mismatch: Hostname differs. (baseline: bench-a; candidate: bench-b)" in text


def test_render_compare_text_warning_lines_render_absent_values_as_unavailable(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_upgraded_session(baseline_dir)
    _write_upgraded_session(candidate_dir)
    candidate_plan = json.loads((candidate_dir / "plan.json").read_text(encoding="utf-8"))
    candidate_plan["execution_settings"]["top_p"] = 0.9
    _write_json(candidate_dir / "plan.json", candidate_plan)

    result = compare_sessions(baseline_dir=baseline_dir, candidate_dir=candidate_dir)
    text = render_compare_text(result)

    assert "execution_policy.top_p_mismatch" in [warning.code for warning in result.warnings]
    assert "(baseline: unavailable; candidate: 0.9)" in text
    assert "(baseline: n/a; candidate: 0.9)" not in text


def test_render_compare_json_exposes_structured_result(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_upgraded_session(baseline_dir)
    _write_upgraded_session(candidate_dir)

    payload = json.loads(render_compare_json(compare_sessions(baseline_dir=baseline_dir, candidate_dir=candidate_dir)))

    assert payload["baseline_path"] == str(baseline_dir)
    assert payload["baseline_methodology_version"] == BENCHMARK_METHODOLOGY_VERSION
    assert payload["candidate_methodology_version"] == BENCHMARK_METHODOLOGY_VERSION
    assert payload["comparability_status"] == "pass"
    assert payload["gates"]["methodology_compatibility"] == "pass"
    assert payload["row_comparisons"][0]["key"]["scenario_id"] == "smoke-basic-v1"
