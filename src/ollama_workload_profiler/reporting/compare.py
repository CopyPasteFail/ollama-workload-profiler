from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict


class CompareGateStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class WarningSeverity(StrEnum):
    INFO = "info"
    MINOR = "minor"
    MAJOR = "major"
    FATAL = "fatal"


class WarningScope(StrEnum):
    METHODOLOGY = "methodology"
    ARTIFACT_SCHEMA = "artifact_schema"
    EXECUTION_POLICY = "execution_policy"
    ENVIRONMENT = "environment"
    ROW_COVERAGE = "row_coverage"
    METRIC = "metric"


class MissingMetricReason(StrEnum):
    AVAILABLE = "available"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"
    INELIGIBLE = "ineligible"
    SCHEMA_UNAVAILABLE = "schema_unavailable"


class CompareWarning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    severity: WarningSeverity
    message: str
    scope: WarningScope
    field: str | None = None
    baseline: Any | None = None
    candidate: Any | None = None


class RowKey(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str
    context_size: int
    benchmark_type: str
    scenario_id: str

    def as_sort_tuple(self) -> tuple[str, int, str, str]:
        return (
            self.model_name,
            self.context_size,
            self.benchmark_type,
            self.scenario_id,
        )

    def display(self) -> str:
        return " | ".join(str(part) for part in self.as_sort_tuple())


class MetricDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: str
    metric: str
    baseline: float | None = None
    candidate: float | None = None
    delta: float | None = None
    delta_percent: float | None = None
    reason: MissingMetricReason


class LoadedCompareSession(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    path: Path
    summary: dict[str, Any]
    plan: dict[str, Any]
    environment: dict[str, Any]
    methodology_version: str | None = None
    methodology_warnings: list[CompareWarning] = []


class MatchedBenchmarkRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: RowKey
    baseline: dict[str, Any]
    candidate: dict[str, Any]


class RowMatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matched: list[MatchedBenchmarkRow]
    only_in_baseline: list[RowKey]
    only_in_candidate: list[RowKey]
    warnings: list[CompareWarning]


class RowComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: RowKey
    metric_deltas: list[MetricDelta]


class CompareResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_path: str
    candidate_path: str
    baseline_methodology_version: str | None = None
    candidate_methodology_version: str | None = None
    comparability_status: CompareGateStatus
    gates: dict[str, CompareGateStatus]
    warnings: list[CompareWarning]
    session_summary_deltas: list[MetricDelta]
    row_comparisons: list[RowComparison]
    only_in_baseline: list[RowKey]
    only_in_candidate: list[RowKey]
    strict_failure_reasons: list[str]


class CompareArtifactError(ValueError):
    pass


REQUIRED_ARTIFACTS = ("summary.json", "plan.json", "environment.json")
EXPECTED_EXECUTION_SETTING_KEYS = {
    "seed",
    "temperature",
    "top_p",
    "repetitions",
    "warmup_runs",
    "warmup_enabled",
}
EXECUTION_POLICY_FIELDS = (
    "seed",
    "temperature",
    "top_p",
    "repetitions",
    "warmup_runs",
    "warmup_enabled",
)
ROW_METRICS = (
    "elapsed_ms_median",
    "elapsed_ms_p95",
    "prompt_tokens_per_second_median",
    "prompt_tokens_per_second_p95",
    "generation_tokens_per_second_median",
    "generation_tokens_per_second_p95",
    "load_duration_ms_median",
    "load_duration_ms_p95",
    "ttft_ms_median",
    "ttft_ms_p95",
)
SESSION_METRICS = (
    "run_count",
    "completed_runs",
    "failed_runs",
    "stopped_runs",
    "completed_sample_size",
    *ROW_METRICS,
)


def load_compare_session(session_dir: Path) -> LoadedCompareSession:
    path = Path(session_dir)
    if not path.exists() or not path.is_dir():
        raise CompareArtifactError(f"Session directory does not exist: {path}")

    artifacts = {artifact_name: _read_json_object(path / artifact_name) for artifact_name in REQUIRED_ARTIFACTS}
    summary = artifacts["summary.json"]
    benchmark_summaries = summary.get("benchmark_summaries")
    if not isinstance(benchmark_summaries, list):
        raise CompareArtifactError("summary.json must contain benchmark_summaries as a list")

    methodology_version, methodology_warnings = _assess_session_methodology_artifacts(
        plan=artifacts["plan.json"],
        environment=artifacts["environment.json"],
        summary=summary,
    )

    return LoadedCompareSession(
        path=path,
        summary=summary,
        plan=artifacts["plan.json"],
        environment=artifacts["environment.json"],
        methodology_version=methodology_version,
        methodology_warnings=methodology_warnings,
    )


def compare_sessions(*, baseline_dir: Path, candidate_dir: Path) -> CompareResult:
    baseline = load_compare_session(baseline_dir)
    candidate = load_compare_session(candidate_dir)

    methodology_gate, methodology_warnings = assess_methodology_compatibility(
        baseline=baseline,
        candidate=candidate,
    )
    execution_gate, execution_warnings = assess_execution_policy_compatibility(
        baseline=baseline,
        candidate=candidate,
    )
    environment_gate, environment_warnings = assess_environment_runtime_comparability(
        baseline=baseline,
        candidate=candidate,
    )
    row_matches = match_benchmark_rows(baseline=baseline, candidate=candidate)

    gates = {
        "methodology_compatibility": methodology_gate,
        "execution_policy_compatibility": execution_gate,
        "artifact_schema_compatibility": CompareGateStatus.PASS,
        "environment_runtime_comparability": environment_gate,
    }
    warnings = [
        *baseline.methodology_warnings,
        *candidate.methodology_warnings,
        *methodology_warnings,
        *execution_warnings,
        *environment_warnings,
        *row_matches.warnings,
    ]
    unmatched_row_count = len(row_matches.only_in_baseline) + len(row_matches.only_in_candidate)
    return CompareResult(
        baseline_path=str(baseline.path),
        candidate_path=str(candidate.path),
        baseline_methodology_version=_explicit_methodology_version(baseline),
        candidate_methodology_version=_explicit_methodology_version(candidate),
        comparability_status=_overall_status(gates, warnings),
        gates=gates,
        warnings=warnings,
        session_summary_deltas=build_session_metric_deltas(
            baseline=_session_metrics(baseline),
            candidate=_session_metrics(candidate),
        ),
        row_comparisons=[
            RowComparison(
                key=matched.key,
                metric_deltas=build_row_metric_deltas(
                    baseline=matched.baseline,
                    candidate=matched.candidate,
                ),
            )
            for matched in row_matches.matched
        ],
        only_in_baseline=row_matches.only_in_baseline,
        only_in_candidate=row_matches.only_in_candidate,
        strict_failure_reasons=strict_mode_failure_reasons(
            warnings=warnings,
            gates=gates,
            unmatched_row_count=unmatched_row_count,
        ),
    )


def render_compare_json(result: CompareResult) -> str:
    return json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def render_compare_text(result: CompareResult, *, all_metrics: bool = False) -> str:
    lines = [
        "Comparison: baseline -> candidate",
        f"Baseline: {result.baseline_path}",
        f"Candidate: {result.candidate_path}",
        (
            "Benchmark methodology: "
            f"baseline {result.baseline_methodology_version or 'unavailable'} -> "
            f"candidate {result.candidate_methodology_version or 'unavailable'}"
        ),
        "",
        f"Comparability: {result.comparability_status.value}",
        f"Methodology: {result.gates['methodology_compatibility'].value}",
        f"Execution policy: {result.gates['execution_policy_compatibility'].value}",
        f"Artifacts/schema: {result.gates['artifact_schema_compatibility'].value}",
        f"Environment/runtime: {result.gates['environment_runtime_comparability'].value}",
        "",
        "Result Summary:",
    ]
    lines.extend(_render_result_summary(result))
    lines.extend(
        [
            "",
            "Warnings:",
        ]
    )
    if result.warnings:
        for warning in result.warnings:
            lines.append(f"- {_format_warning(warning)}")
    else:
        lines.append("- none")

    lines.extend(["", "Session Summary:"])
    lines.extend(_render_delta_table(result.session_summary_deltas, all_metrics=all_metrics))

    lines.extend(["", "Rows:"])
    if result.row_comparisons:
        unchanged_row_count = 0
        for row in result.row_comparisons:
            if not all_metrics and not _changed_deltas(row.metric_deltas):
                unchanged_row_count += 1
                continue
            lines.append(
                "model="
                f"{row.key.model_name} ctx={row.key.context_size} "
                f"benchmark={row.key.benchmark_type} scenario={row.key.scenario_id}"
            )
            lines.extend(_render_delta_table(row.metric_deltas, all_metrics=all_metrics))
        if not all_metrics:
            lines.append(f"Rows without changed metrics: {unchanged_row_count}")
    else:
        lines.append("No matched rows.")

    lines.extend(["", "Unmatched Rows:", "Only in baseline:"])
    lines.extend([f"- {key.display()}" for key in result.only_in_baseline] or ["- none"])
    lines.append("Only in candidate:")
    lines.extend([f"- {key.display()}" for key in result.only_in_candidate] or ["- none"])

    return "\n".join(lines).rstrip() + "\n"


def _render_result_summary(result: CompareResult) -> list[str]:
    lines = [
        f"- Matched rows: {len(result.row_comparisons)}",
        f"- Warnings: {_format_warning_counts(result.warnings)}",
    ]
    notable_deltas = _changed_deltas(result.session_summary_deltas)[:5]
    if notable_deltas:
        lines.append("- Notable session deltas:")
        for delta in notable_deltas:
            lines.append(f"  - {_format_delta_summary(delta)}")
    else:
        lines.append("- Notable session deltas: none")
    return lines


def _format_warning_counts(warnings: list[CompareWarning]) -> str:
    if not warnings:
        return "none"
    counts = {
        severity: sum(1 for warning in warnings if warning.severity is severity)
        for severity in WarningSeverity
    }
    return ", ".join(
        f"{severity.value}={count}"
        for severity, count in counts.items()
        if count
    )


def _format_delta_summary(delta: MetricDelta) -> str:
    return (
        f"{delta.metric}: {_format_metric_value(delta.metric, delta.baseline)} -> "
        f"{_format_metric_value(delta.metric, delta.candidate)} "
        f"(delta {_format_metric_value(delta.metric, delta.delta)}, {_format_percent(delta.delta_percent)})"
    )


def _format_warning(warning: CompareWarning) -> str:
    value_suffix = ""
    if warning.baseline is not None or warning.candidate is not None:
        value_suffix = (
            f" (baseline: {_format_warning_value(warning.baseline)}; "
            f"candidate: {_format_warning_value(warning.candidate)})"
        )
    return f"[{warning.severity.value}] {warning.code}: {warning.message}{value_suffix}"


def _format_warning_value(value: object) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _session_metrics(session: LoadedCompareSession) -> dict[str, Any]:
    value = session.summary.get("session_metrics")
    return dict(value) if isinstance(value, dict) else {}


def _overall_status(
    gates: Mapping[str, CompareGateStatus],
    warnings: list[CompareWarning],
) -> CompareGateStatus:
    if any(gate is CompareGateStatus.FAIL for gate in gates.values()):
        return CompareGateStatus.FAIL
    if any(warning.severity is WarningSeverity.FATAL for warning in warnings):
        return CompareGateStatus.FAIL
    if any(gate is CompareGateStatus.WARN for gate in gates.values()) or warnings:
        return CompareGateStatus.WARN
    return CompareGateStatus.PASS


def _render_delta_table(deltas: list[MetricDelta], *, all_metrics: bool) -> list[str]:
    rendered_deltas = deltas if all_metrics else _changed_deltas(deltas)
    if not rendered_deltas:
        return ["No changed metrics."]

    lines = ["| metric | baseline | candidate | delta | delta % | reason |", "| --- | --- | --- | --- | --- | --- |"]
    for delta in rendered_deltas:
        lines.append(
            "| "
            + " | ".join(
                (
                    delta.metric,
                    _format_metric_value(delta.metric, delta.baseline),
                    _format_metric_value(delta.metric, delta.candidate),
                    _format_metric_value(delta.metric, delta.delta),
                    _format_percent(delta.delta_percent),
                    delta.reason.value,
                )
            )
            + " |"
        )
    return lines


def _changed_deltas(deltas: list[MetricDelta]) -> list[MetricDelta]:
    return [
        delta
        for delta in deltas
        if delta.reason is MissingMetricReason.AVAILABLE
        and delta.delta is not None
        and delta.delta != 0
    ]


def _format_metric_value(metric: str, value: float | None) -> str:
    if value is None:
        return "n/a"
    formatted = _format_number(value)
    if metric.startswith(("elapsed_ms", "load_duration_ms", "ttft_ms")):
        return f"{formatted} ms"
    if "tokens_per_second" in metric:
        return f"{formatted} tok/s"
    return formatted


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{_format_number(value)}%"


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise CompareArtifactError(f"Missing required artifact: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CompareArtifactError(f"Malformed JSON artifact: {path.name}") from exc
    if not isinstance(payload, dict):
        raise CompareArtifactError(f"{path.name} must contain a JSON object")
    return payload


def _assess_session_methodology_artifacts(
    *,
    plan: Mapping[str, Any],
    environment: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> tuple[str | None, list[CompareWarning]]:
    artifact_versions = {
        "plan.json": _artifact_methodology_version(plan),
        "environment.json": _artifact_methodology_version(environment),
        "summary.json": _artifact_methodology_version(summary),
    }
    present_versions = {
        artifact_name: version
        for artifact_name, version in artifact_versions.items()
        if version is not None
    }
    effective_version = _preferred_methodology_version(present_versions)

    if len(set(present_versions.values())) > 1:
        return (
            effective_version,
            [
                CompareWarning(
                    code="methodology.artifact_version_conflict",
                    severity=WarningSeverity.MAJOR,
                    message="Benchmark methodology versions conflict within one session.",
                    scope=WarningScope.METHODOLOGY,
                    field="benchmark_methodology_version",
                    baseline=dict(sorted(present_versions.items())),
                )
            ],
        )

    missing_artifacts = [
        artifact_name
        for artifact_name, version in artifact_versions.items()
        if version is None
    ]
    if missing_artifacts and present_versions:
        missing_field = (
            f"{missing_artifacts[0]}.benchmark_methodology_version"
            if len(missing_artifacts) == 1
            else "benchmark_methodology_version"
        )
        return (
            effective_version,
            [
                CompareWarning(
                    code="methodology.artifact_version_missing",
                    severity=WarningSeverity.MINOR,
                    message="Benchmark methodology version is missing from one or more session artifacts.",
                    scope=WarningScope.METHODOLOGY,
                    field=missing_field,
                    baseline=missing_artifacts,
                    candidate=effective_version,
                )
            ],
        )

    return effective_version, []


def _artifact_methodology_version(artifact: Mapping[str, Any]) -> str | None:
    value = artifact.get("benchmark_methodology_version")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _preferred_methodology_version(versions: Mapping[str, str]) -> str | None:
    for artifact_name in ("plan.json", "environment.json", "summary.json"):
        version = versions.get(artifact_name)
        if version is not None:
            return version
    return None


def assess_methodology_compatibility(
    *,
    baseline: LoadedCompareSession,
    candidate: LoadedCompareSession,
) -> tuple[CompareGateStatus, list[CompareWarning]]:
    baseline_version = _explicit_methodology_version(baseline)
    candidate_version = _explicit_methodology_version(candidate)
    if baseline_version and candidate_version:
        if baseline_version == candidate_version:
            return CompareGateStatus.PASS, []
        return (
            CompareGateStatus.WARN,
            [
                CompareWarning(
                    code="methodology.version_mismatch",
                    severity=WarningSeverity.MAJOR,
                    message="Benchmark methodology versions differ.",
                    scope=WarningScope.METHODOLOGY,
                    field="benchmark_methodology_version",
                    baseline=baseline_version,
                    candidate=candidate_version,
                )
            ],
        )

    baseline_upgraded = _has_upgraded_benchmark_contract(baseline)
    candidate_upgraded = _has_upgraded_benchmark_contract(candidate)
    if baseline_upgraded and candidate_upgraded:
        return CompareGateStatus.PASS, []

    return (
        CompareGateStatus.WARN,
        [
            CompareWarning(
                code="methodology.contract_unavailable",
                severity=WarningSeverity.MAJOR,
                message=(
                    "These sessions are not directly comparable under the same "
                    "benchmark contract."
                ),
                scope=WarningScope.METHODOLOGY,
                field="benchmark_contract",
                baseline="upgraded" if baseline_upgraded else "unavailable",
                candidate="upgraded" if candidate_upgraded else "unavailable",
            )
        ],
    )


def _explicit_methodology_version(session: LoadedCompareSession) -> str | None:
    return session.methodology_version


def _has_upgraded_benchmark_contract(session: LoadedCompareSession) -> bool:
    plan_settings = session.plan.get("execution_settings")
    environment_settings = session.environment.get("execution_settings")
    if not _has_expected_execution_settings(plan_settings):
        return False
    if not _has_expected_execution_settings(environment_settings):
        return False

    benchmark_summaries = session.summary.get("benchmark_summaries")
    if not isinstance(benchmark_summaries, list) or not benchmark_summaries:
        return False

    return any(_row_has_upgraded_aggregate_contract(row) for row in benchmark_summaries)


def _has_expected_execution_settings(value: Any) -> bool:
    return isinstance(value, dict) and EXPECTED_EXECUTION_SETTING_KEYS.issubset(value.keys())


def _row_has_upgraded_aggregate_contract(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    has_strict_sample = "strict_sample_size" in row
    has_median = any(str(key).endswith("_median") for key in row)
    has_p95 = any(str(key).endswith("_p95") for key in row)
    return has_strict_sample and has_median and has_p95


def assess_execution_policy_compatibility(
    *,
    baseline: LoadedCompareSession,
    candidate: LoadedCompareSession,
) -> tuple[CompareGateStatus, list[CompareWarning]]:
    baseline_settings = baseline.plan.get("execution_settings")
    candidate_settings = candidate.plan.get("execution_settings")
    if not isinstance(baseline_settings, dict) or not isinstance(candidate_settings, dict):
        return (
            CompareGateStatus.WARN,
            [
                CompareWarning(
                    code="execution_policy.execution_settings_unavailable",
                    severity=WarningSeverity.MAJOR,
                    message="Execution settings are unavailable in one or both sessions.",
                    scope=WarningScope.EXECUTION_POLICY,
                    field="execution_settings",
                    baseline="available" if isinstance(baseline_settings, dict) else "unavailable",
                    candidate="available" if isinstance(candidate_settings, dict) else "unavailable",
                )
            ],
        )

    warnings: list[CompareWarning] = []
    for field in EXECUTION_POLICY_FIELDS:
        baseline_value = baseline_settings.get(field)
        candidate_value = candidate_settings.get(field)
        if baseline_value == candidate_value:
            continue
        warnings.append(
            CompareWarning(
                code=f"execution_policy.{field}_mismatch",
                severity=WarningSeverity.MAJOR,
                message=f"Execution setting {field} differs.",
                scope=WarningScope.EXECUTION_POLICY,
                field=f"execution_settings.{field}",
                baseline=baseline_value,
                candidate=candidate_value,
            )
        )

    return (CompareGateStatus.WARN if warnings else CompareGateStatus.PASS, warnings)


def assess_environment_runtime_comparability(
    *,
    baseline: LoadedCompareSession,
    candidate: LoadedCompareSession,
) -> tuple[CompareGateStatus, list[CompareWarning]]:
    checks = (
        (
            "environment.ollama_version_mismatch",
            WarningSeverity.MAJOR,
            "Ollama version differs.",
            "ollama.version.value",
        ),
        (
            "environment.model_details_mismatch",
            WarningSeverity.MAJOR,
            "Selected model metadata differs.",
            "ollama.selected_model.details",
        ),
        (
            "environment.accelerator_mismatch",
            WarningSeverity.MAJOR,
            "Accelerator metadata differs.",
            "accelerator",
        ),
        (
            "environment.hostname_mismatch",
            WarningSeverity.MINOR,
            "Hostname differs.",
            "host.hostname",
        ),
        (
            "environment.python_version_mismatch",
            WarningSeverity.MINOR,
            "Python version differs.",
            "python_version",
        ),
    )
    warnings: list[CompareWarning] = []
    for code, severity, message, field in checks:
        baseline_value = _get_dotted_value(baseline.environment, field)
        candidate_value = _get_dotted_value(candidate.environment, field)
        if baseline_value == candidate_value:
            continue
        if baseline_value is None and candidate_value is None:
            continue
        warnings.append(
            CompareWarning(
                code=code,
                severity=severity,
                message=message,
                scope=WarningScope.ENVIRONMENT,
                field=field,
                baseline=baseline_value,
                candidate=candidate_value,
            )
        )

    return (CompareGateStatus.WARN if warnings else CompareGateStatus.PASS, warnings)


def _get_dotted_value(payload: Mapping[str, Any], field: str) -> Any:
    current: Any = payload
    for part in field.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def match_benchmark_rows(
    *,
    baseline: LoadedCompareSession,
    candidate: LoadedCompareSession,
) -> RowMatchResult:
    baseline_rows = _index_benchmark_summary_rows(baseline)
    candidate_rows = _index_benchmark_summary_rows(candidate)
    baseline_keys = set(baseline_rows)
    candidate_keys = set(candidate_rows)

    matched_keys = sorted(baseline_keys & candidate_keys, key=lambda key: key.as_sort_tuple())
    only_in_baseline = sorted(baseline_keys - candidate_keys, key=lambda key: key.as_sort_tuple())
    only_in_candidate = sorted(candidate_keys - baseline_keys, key=lambda key: key.as_sort_tuple())

    warnings: list[CompareWarning] = []
    unmatched_count = len(only_in_baseline) + len(only_in_candidate)
    if unmatched_count:
        warnings.append(
            CompareWarning(
                code="row_coverage.unmatched_rows",
                severity=WarningSeverity.MAJOR,
                message="Some benchmark summary rows are present in only one session.",
                scope=WarningScope.ROW_COVERAGE,
                field="benchmark_summaries",
                baseline=len(only_in_baseline),
                candidate=len(only_in_candidate),
            )
        )

    return RowMatchResult(
        matched=[
            MatchedBenchmarkRow(
                key=key,
                baseline=baseline_rows[key],
                candidate=candidate_rows[key],
            )
            for key in matched_keys
        ],
        only_in_baseline=only_in_baseline,
        only_in_candidate=only_in_candidate,
        warnings=warnings,
    )


def build_row_metric_deltas(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> list[MetricDelta]:
    return [_build_metric_delta(scope="row", metric=metric, baseline=baseline, candidate=candidate) for metric in ROW_METRICS]


def build_session_metric_deltas(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> list[MetricDelta]:
    return [
        _build_metric_delta(scope="session", metric=metric, baseline=baseline, candidate=candidate)
        for metric in SESSION_METRICS
    ]


def _build_metric_delta(
    *,
    scope: str,
    metric: str,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> MetricDelta:
    baseline_value = _numeric_value(baseline.get(metric))
    candidate_value = _numeric_value(candidate.get(metric))
    reason = _metric_reason(metric=metric, baseline=baseline, candidate=candidate)
    if reason is not MissingMetricReason.AVAILABLE:
        return MetricDelta(
            scope=scope,
            metric=metric,
            baseline=baseline_value,
            candidate=candidate_value,
            reason=reason,
        )

    delta = round(candidate_value - baseline_value, 3) if baseline_value is not None and candidate_value is not None else None
    delta_percent: float | None = None
    if baseline_value not in (None, 0.0) and delta is not None:
        delta_percent = round((delta / baseline_value) * 100, 3)
    return MetricDelta(
        scope=scope,
        metric=metric,
        baseline=baseline_value,
        candidate=candidate_value,
        delta=delta,
        delta_percent=delta_percent,
        reason=MissingMetricReason.AVAILABLE,
    )


def _metric_reason(
    *,
    metric: str,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> MissingMetricReason:
    baseline_has_metric = metric in baseline
    candidate_has_metric = metric in candidate
    if baseline_has_metric and candidate_has_metric:
        if _numeric_value(baseline.get(metric)) is not None and _numeric_value(candidate.get(metric)) is not None:
            return MissingMetricReason.AVAILABLE
        return MissingMetricReason.MISSING

    if _metric_sample_size(metric, baseline) == 0 or _metric_sample_size(metric, candidate) == 0:
        return MissingMetricReason.INELIGIBLE

    if metric.startswith("ttft_ms") and not baseline_has_metric and not candidate_has_metric:
        return MissingMetricReason.NOT_APPLICABLE

    if _looks_like_legacy_metric_schema(baseline) or _looks_like_legacy_metric_schema(candidate):
        return MissingMetricReason.SCHEMA_UNAVAILABLE

    return MissingMetricReason.MISSING


def _metric_sample_size(metric: str, row: Mapping[str, Any]) -> int | None:
    sample_key = _sample_size_key(metric)
    value = row.get(sample_key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _sample_size_key(metric: str) -> str:
    if metric.endswith("_median"):
        return metric.removesuffix("_median") + "_sample_size"
    if metric.endswith("_p95"):
        return metric.removesuffix("_p95") + "_sample_size"
    return metric + "_sample_size"


def _looks_like_legacy_metric_schema(row: Mapping[str, Any]) -> bool:
    metric_keys = [str(key) for key in row]
    has_aggregate_metric = any(key.endswith("_median") or key.endswith("_p95") for key in metric_keys)
    has_aggregate_sample = any(key.endswith("_sample_size") for key in metric_keys)
    return bool(metric_keys) and not has_aggregate_metric and not has_aggregate_sample


def _numeric_value(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return float(value)


def _index_benchmark_summary_rows(session: LoadedCompareSession) -> dict[RowKey, dict[str, Any]]:
    rows: dict[RowKey, dict[str, Any]] = {}
    for raw_row in session.summary["benchmark_summaries"]:
        if not isinstance(raw_row, dict):
            raise CompareArtifactError("benchmark_summaries rows must be JSON objects")
        key = _row_key_from_summary_row(raw_row)
        if key in rows:
            raise CompareArtifactError(f"Duplicate benchmark summary row key: {key.display()}")
        rows[key] = dict(raw_row)
    return rows


def _row_key_from_summary_row(row: Mapping[str, Any]) -> RowKey:
    try:
        model_name = row["model_name"]
        context_size = row["context_size"]
        benchmark_type = row["benchmark_type"]
        scenario_id = row["scenario_id"]
    except KeyError as exc:
        raise CompareArtifactError(f"Missing benchmark summary row key field: {exc.args[0]}") from exc
    if not isinstance(context_size, int) or isinstance(context_size, bool):
        raise CompareArtifactError("benchmark summary row context_size must be an integer")
    return RowKey(
        model_name=str(model_name),
        context_size=context_size,
        benchmark_type=str(benchmark_type),
        scenario_id=str(scenario_id),
    )


STRICT_FAIL_SCOPES = {
    WarningScope.METHODOLOGY,
    WarningScope.ARTIFACT_SCHEMA,
    WarningScope.EXECUTION_POLICY,
    WarningScope.ROW_COVERAGE,
}


def strict_mode_failure_reasons(
    *,
    warnings: list[CompareWarning],
    gates: Mapping[str, CompareGateStatus],
    unmatched_row_count: int,
) -> list[str]:
    reasons: list[str] = []
    for warning in warnings:
        if warning.severity is WarningSeverity.FATAL:
            reasons.append(warning.code)
            continue
        if warning.severity is WarningSeverity.MAJOR and warning.scope in STRICT_FAIL_SCOPES:
            reasons.append(warning.code)

    if unmatched_row_count and "row_coverage.unmatched_rows" not in reasons:
        reasons.append("row_coverage.unmatched_rows")

    for gate_name, gate_status in gates.items():
        if gate_status is CompareGateStatus.FAIL and gate_name not in reasons:
            reasons.append(gate_name)

    return reasons
