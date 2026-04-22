from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from math import ceil
from statistics import median
from typing import Any, Mapping, Sequence

from pydantic import BaseModel

from ..methodology import BENCHMARK_METHODOLOGY_VERSION
from ..models.failures import FailureInfo
from ..models.plan import BenchmarkType
from ..models.results import RunResult
from ..models.summary import ModelSummary, ReportSummary
from ..prompts.scenarios import build_scenarios_for_benchmark
from .verdicts import build_missing_use_case_cell, summarize_use_case_cell


SUMMARY_METRIC_NAMES: tuple[str, ...] = (
    "elapsed_ms",
    "ttft_ms",
    "stream_emission_count",
    "stream_duration_ms",
    "stream_emission_interval_ms_median",
    "stream_emission_interval_ms_p95",
    "stream_output_units_per_second",
    "concurrency_request_elapsed_ms_p50",
    "concurrency_request_elapsed_ms_p95",
    "concurrency_request_ttft_ms_p50",
    "concurrency_request_ttft_ms_p95",
    "available_system_memory_mb",
    "system_cpu_load_snapshot",
    "peak_gpu_memory_mb",
    "avg_gpu_util_percent",
    "peak_gpu_util_percent",
    "prompt_eval_count",
    "actual_prompt_tokens",
    "target_prompt_tokens",
    "prompt_tokens_per_second",
    "generation_tokens_per_second",
    "load_duration_ms",
)

SUMMARY_METRIC_DISPLAY_NAMES: dict[str, str] = {
    "elapsed_ms": "run elapsed time",
    "ttft_ms": "TTFT",
    "prompt_tokens_per_second": "prompt tokens/s",
    "generation_tokens_per_second": "generation tokens/s",
    "load_duration_ms": "load duration",
}


def build_report_summary(
    *,
    plan: Mapping[str, Any] | BaseModel,
    environment: Mapping[str, Any] | BaseModel,
    runs: Sequence[RunResult | Mapping[str, Any] | BaseModel],
) -> ReportSummary:
    plan_payload = _to_json_payload(plan)
    environment_payload = _to_json_payload(environment)
    run_payloads = [_to_json_payload(run) for run in runs]

    session_metrics = _build_session_metrics(plan_payload, run_payloads)
    model_summaries = _build_model_summaries(run_payloads)
    benchmark_summaries = _build_benchmark_summaries(run_payloads)
    phase_peak_summaries = _build_phase_peak_summaries(run_payloads)
    use_case_matrix = _build_use_case_matrix(plan_payload, run_payloads)
    failures = _build_failures(run_payloads)
    warnings = _build_warnings(session_metrics, run_payloads)
    recommendations = _build_recommendations(session_metrics)

    return ReportSummary(
        benchmark_methodology_version=_methodology_version_from_artifacts(
            plan_payload=plan_payload,
            environment_payload=environment_payload,
        ),
        executive_summary=_build_executive_summary(session_metrics, plan_payload),
        session_metrics=session_metrics,
        model_summaries=model_summaries,
        use_case_matrix=use_case_matrix,
        benchmark_summaries=benchmark_summaries,
        phase_peak_summaries=phase_peak_summaries,
        warnings=warnings,
        failures=failures,
        recommendations=recommendations,
    )


def _methodology_version_from_artifacts(
    *,
    plan_payload: Mapping[str, Any],
    environment_payload: Mapping[str, Any],
) -> str | None:
    for payload in (plan_payload, environment_payload):
        value = payload.get("benchmark_methodology_version")
        if isinstance(value, str) and value.strip():
            return value
    if plan_payload or environment_payload:
        return BENCHMARK_METHODOLOGY_VERSION
    return None


def _to_json_payload(value: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return deepcopy(value.model_dump(mode="json"))
    return deepcopy(dict(value))


def _build_session_metrics(plan_payload: dict[str, Any], run_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    state_counts: dict[str, int] = defaultdict(int)
    completed_runs: list[dict[str, Any]] = []

    for run in run_payloads:
        state = str(run.get("state", "unknown"))
        state_counts[state] += 1

        if state == "completed":
            completed_runs.append(run)

    metrics = {
        "planned_run_count": len(run_payloads) or None,
        "run_count": len(run_payloads),
        "completed_runs": state_counts.get("completed", 0),
        "failed_runs": state_counts.get("failed", 0),
        "stopped_runs": state_counts.get("stopped", 0),
        "planned_runs": state_counts.get("planned", 0),
        "starting_runs": state_counts.get("starting", 0),
        "model_count": len({run.get("model_name") for run in run_payloads if run.get("model_name")}),
        "context_count": len({run.get("context_size") for run in run_payloads if run.get("context_size") is not None}),
        "benchmark_count": len({run.get("benchmark_type") for run in run_payloads if run.get("benchmark_type")}),
        "completed_sample_size": len(completed_runs),
    }
    for metric_name in SUMMARY_METRIC_NAMES:
        metrics.update(
            _build_metric_summary(
                completed_runs,
                metric_name,
                source=_summary_metric_source(metric_name),
            )
        )
    return metrics


def _build_executive_summary(session_metrics: dict[str, Any], plan_payload: dict[str, Any]) -> str:
    completed_runs = int(session_metrics.get("completed_runs") or 0)
    run_count = int(session_metrics.get("run_count") or 0)
    model_name = plan_payload.get("model_name")

    if completed_runs == 0:
        return "No completed runs were recorded."

    model_fragment = f" for {model_name}" if model_name else ""
    headline_parts = [
        f"{completed_runs} of {run_count} runs completed successfully{model_fragment}.",
        _format_metric_headline(session_metrics, "elapsed_ms"),
        _format_metric_headline(session_metrics, "ttft_ms"),
        _format_metric_headline(session_metrics, "prompt_tokens_per_second"),
        _format_metric_headline(session_metrics, "generation_tokens_per_second"),
        _format_metric_headline(session_metrics, "load_duration_ms"),
    ]
    return " ".join(part for part in headline_parts if part)


def _build_model_summaries(run_payloads: list[dict[str, Any]]) -> dict[str, ModelSummary]:
    grouped_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in run_payloads:
        model_name = str(run.get("model_name") or "unknown")
        grouped_runs[model_name].append(run)

    model_summaries: dict[str, ModelSummary] = {}
    for model_name, grouped in grouped_runs.items():
        completed_runs = [run for run in grouped if run.get("state") == "completed"]
        metric_summary: dict[str, Any] = {
            "run_count": len(grouped),
            "completed_runs": len(completed_runs),
            "failed_runs": sum(1 for run in grouped if run.get("state") == "failed"),
            "completed_sample_size": len(completed_runs),
        }
        for metric_name in SUMMARY_METRIC_NAMES:
            metric_summary.update(
                _build_metric_summary(
                    completed_runs,
                    metric_name,
                    source=_summary_metric_source(metric_name),
                )
            )
        model_summaries[model_name] = ModelSummary(
            summary=(
                "No completed runs were recorded."
                if not completed_runs
                else (
                    f"{len(completed_runs)} completed sample(s) out of {len(grouped)} run(s). "
                    f"{_format_metric_headline(metric_summary, 'elapsed_ms')}"
                ).strip()
            ),
            metrics=metric_summary,
            notes=_model_notes(grouped),
        )
    return model_summaries


def _model_notes(grouped_runs: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    if any(run.get("state") == "failed" for run in grouped_runs):
        notes.append("At least one run failed.")
    if any(run.get("state") == "stopped" for run in grouped_runs):
        notes.append("At least one run stopped early.")
    return notes


def _build_benchmark_summaries(run_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_runs: dict[tuple[Any, Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for run in run_payloads:
        key = (
            run.get("model_name"),
            run.get("context_size"),
            run.get("benchmark_type"),
            run.get("scenario_id"),
        )
        grouped_runs[key].append(run)

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped_runs, key=lambda item: tuple("" if part is None else str(part) for part in item)):
        grouped = grouped_runs[key]
        completed_runs = [run for run in grouped if run.get("state") == "completed"]
        strict_runs = [
            run
            for run in completed_runs
            if _metric_bool(run, "eligible_for_strict_aggregate")
        ]
        rows.append(
            {
                "model_name": key[0],
                "context_size": key[1],
                "benchmark_type": key[2],
                "scenario_id": key[3],
                "sample_size": len(grouped),
                "completed_runs": len(completed_runs),
                "failed_runs": sum(1 for run in grouped if run.get("state") == "failed"),
                "stopped_runs": sum(1 for run in grouped if run.get("state") == "stopped"),
                "strict_sample_size": len(strict_runs),
                **_build_benchmark_metric_summaries(strict_runs),
            }
        )
    return rows


def _build_phase_peak_summaries(run_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phase_aggregates: dict[str, dict[str, float]] = {}

    for run in run_payloads:
        metrics = run.get("metrics")
        if not isinstance(metrics, dict):
            continue
        phase_peaks = metrics.get("phase_peaks")
        if not isinstance(phase_peaks, dict):
            continue

        for phase, peak_values in phase_peaks.items():
            if not isinstance(peak_values, dict):
                continue
            aggregate = phase_aggregates.setdefault(str(phase), {"rss_mb": 0.0, "cpu_percent": 0.0})
            rss_mb = peak_values.get("rss_mb")
            cpu_percent = peak_values.get("cpu_percent")
            if isinstance(rss_mb, int | float):
                aggregate["rss_mb"] = max(aggregate["rss_mb"], float(rss_mb))
            if isinstance(cpu_percent, int | float):
                aggregate["cpu_percent"] = max(aggregate["cpu_percent"], float(cpu_percent))

    return [
        {
            "phase": phase,
            "peak_rss_mb": round(values["rss_mb"], 3),
            "peak_cpu_percent": round(values["cpu_percent"], 3),
        }
        for phase, values in sorted(phase_aggregates.items())
    ]


def _build_benchmark_metric_summaries(strict_runs: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for metric_name in SUMMARY_METRIC_NAMES:
        summaries.update(
            _build_metric_summary(
                strict_runs,
                metric_name,
                source=_summary_metric_source(metric_name),
                eligibility_flag="eligible_for_ttft_aggregate" if metric_name == "ttft_ms" else None,
            )
        )
    return summaries


def _build_use_case_matrix(
    plan_payload: dict[str, Any],
    run_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    use_case_runs = [
        run
        for run in run_payloads
        if _benchmark_type_value(run.get("benchmark_type")) == BenchmarkType.USE_CASE_PROFILES.value
    ]
    if not use_case_runs:
        return []

    selected_contexts = _selected_contexts(plan_payload, use_case_runs)
    grouped_runs: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    scenario_names = _use_case_scenario_names(selected_contexts)

    for run in use_case_runs:
        scenario_id = _optional_str(run.get("scenario_id")) or "unknown-scenario"
        context_size = run.get("context_size")
        if not isinstance(context_size, int):
            continue
        grouped_runs[(scenario_id, context_size)].append(run)

    rows: list[dict[str, Any]] = []
    for scenario_id, scenario_name in scenario_names.items():
        row: dict[str, Any] = {
            "use_case_profile": scenario_name,
            "scenario_id": scenario_id,
        }
        for context_size in selected_contexts:
            grouped = grouped_runs.get((scenario_id, context_size))
            if grouped:
                row[str(context_size)] = summarize_use_case_cell(runs=grouped)
            else:
                row[str(context_size)] = build_missing_use_case_cell()
        rows.append(row)

    return rows


def _build_failures(run_payloads: list[dict[str, Any]]) -> list[FailureInfo]:
    failures: list[FailureInfo] = []
    for run in run_payloads:
        state = run.get("state")
        if state not in {"failed", "stopped"}:
            continue
        failure_payload = run.get("failure")
        if isinstance(failure_payload, dict):
            failures.append(FailureInfo.model_validate(failure_payload))
            continue

        metrics = run.get("metrics")
        if not isinstance(metrics, dict):
            continue

        message = metrics.get("finalization_error") or metrics.get("stop_reason") or "Run did not complete successfully."
        failures.append(
            FailureInfo(
                kind=str(state),
                phase="execution",
                message=str(message),
                exception_class=_optional_str(metrics.get("exception_class")),
            )
        )
    return failures


def _build_warnings(session_metrics: dict[str, Any], run_payloads: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    if int(session_metrics.get("failed_runs") or 0) > 0:
        warnings.append("One or more runs failed; review raw.jsonl for per-run details.")
    if int(session_metrics.get("stopped_runs") or 0) > 0:
        warnings.append("One or more runs stopped before completion.")
    host_pressure_reasons = _host_pressure_warning_reasons(run_payloads)
    if host_pressure_reasons:
        warnings.append("One or more runs started with advisory host pressure warnings.")
        warnings.append(f"Host pressure reasons observed: {'; '.join(host_pressure_reasons)}")
    if not run_payloads:
        warnings.append("No runs were recorded for this session.")
    return warnings


def _build_recommendations(session_metrics: dict[str, Any]) -> list[str]:
    if int(session_metrics.get("completed_runs") or 0) == 0:
        return ["Run at least one benchmark to generate performance recommendations."]
    return ["Use raw.jsonl for per-run inspection and summary.json for aggregate comparisons."]


def _build_metric_summary(
    runs: Sequence[dict[str, Any]],
    metric_name: str,
    *,
    source: str = "metrics",
    eligibility_flag: str | None = None,
) -> dict[str, Any]:
    values: list[float] = []
    for run in runs:
        if eligibility_flag and not _metric_bool(run, eligibility_flag):
            continue
        value = _numeric_metric_value(run, metric_name, source=source)
        if value is not None:
            values.append(value)

    return {
        f"{metric_name}_sample_size": len(values),
        f"{metric_name}_median": _round_metric(median(values)) if values else None,
        f"{metric_name}_p95": _round_metric(_p95_nearest_rank(values)) if values else None,
    }


def _numeric_metric_value(run: Mapping[str, Any], metric_name: str, *, source: str) -> float | None:
    if source == "run":
        value = run.get(metric_name)
    elif source == "system_snapshot":
        system_snapshot = run.get("system_snapshot")
        if not isinstance(system_snapshot, dict):
            return None
        value = system_snapshot.get(metric_name)
    else:
        metrics = run.get("metrics")
        if not isinstance(metrics, dict):
            return None
        value = metrics.get(metric_name)
        if metric_name == "generation_tokens_per_second" and not isinstance(value, (int, float)):
            value = metrics.get("tokens_per_second")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _summary_metric_source(metric_name: str) -> str:
    if metric_name == "elapsed_ms":
        return "run"
    if metric_name in {"available_system_memory_mb", "system_cpu_load_snapshot"}:
        return "system_snapshot"
    return "metrics"


def _metric_bool(run: Mapping[str, Any], metric_name: str) -> bool:
    metrics = run.get("metrics")
    return isinstance(metrics, dict) and bool(metrics.get(metric_name))


def _host_pressure_warning_reasons(run_payloads: Sequence[Mapping[str, Any]]) -> list[str]:
    reasons: set[str] = set()
    for run in run_payloads:
        system_snapshot = run.get("system_snapshot")
        if not isinstance(system_snapshot, dict):
            continue
        if not system_snapshot.get("host_pressure_warning"):
            continue
        raw_reasons = system_snapshot.get("host_pressure_warning_reasons")
        if not isinstance(raw_reasons, list):
            continue
        reasons.update(reason for reason in raw_reasons if isinstance(reason, str) and reason)
    return sorted(reasons)


def _round_metric(value: float) -> float:
    return round(float(value), 3)


def _p95_nearest_rank(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    rank = max(1, ceil(len(ordered) * 0.95))
    return ordered[rank - 1]


def _format_metric_headline(metrics: Mapping[str, Any], metric_name: str) -> str:
    sample_size = metrics.get(f"{metric_name}_sample_size")
    median_value = metrics.get(f"{metric_name}_median")
    p95_value = metrics.get(f"{metric_name}_p95")
    display_name = SUMMARY_METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    if not sample_size:
        return f"{display_name} unavailable."
    return f"{display_name} median {median_value}, p95 {p95_value} (n={sample_size})."


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _benchmark_type_value(value: object) -> str:
    if hasattr(value, "value"):
        return str(getattr(value, "value"))
    return str(value)


def _selected_contexts(plan_payload: dict[str, Any], use_case_runs: list[dict[str, Any]]) -> list[int]:
    plan_contexts = [context for context in plan_payload.get("contexts", []) if isinstance(context, int)]
    if plan_contexts:
        return list(dict.fromkeys(plan_contexts))

    observed_contexts: list[int] = []
    for run in use_case_runs:
        context_size = run.get("context_size")
        if isinstance(context_size, int) and context_size not in observed_contexts:
            observed_contexts.append(context_size)
    return observed_contexts


def _use_case_scenario_names(selected_contexts: list[int]) -> dict[str, str]:
    scenario_context = selected_contexts[0] if selected_contexts else 4096
    scenarios = build_scenarios_for_benchmark(BenchmarkType.USE_CASE_PROFILES, scenario_context)
    return {scenario.scenario_id: scenario.name for scenario in scenarios}
