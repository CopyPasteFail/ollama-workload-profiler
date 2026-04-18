from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from statistics import mean
from typing import Any, Mapping, Sequence

from pydantic import BaseModel

from ..models.failures import FailureInfo
from ..models.plan import BenchmarkType
from ..models.results import RunResult
from ..models.summary import ModelSummary, ReportSummary
from ..prompts.scenarios import build_scenarios_for_benchmark
from .verdicts import build_missing_use_case_cell, summarize_use_case_cell


def build_report_summary(
    *,
    plan: Mapping[str, Any] | BaseModel,
    environment: Mapping[str, Any] | BaseModel,
    runs: Sequence[RunResult | Mapping[str, Any] | BaseModel],
) -> ReportSummary:
    plan_payload = _to_json_payload(plan)
    _ = _to_json_payload(environment)
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


def _to_json_payload(value: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return deepcopy(value.model_dump(mode="json"))
    return deepcopy(dict(value))


def _build_session_metrics(plan_payload: dict[str, Any], run_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    state_counts: dict[str, int] = defaultdict(int)
    completed_elapsed_ms: list[float] = []

    for run in run_payloads:
        state = str(run.get("state", "unknown"))
        state_counts[state] += 1

        elapsed_ms = run.get("elapsed_ms")
        if state == "completed" and isinstance(elapsed_ms, int | float):
            completed_elapsed_ms.append(float(elapsed_ms))

    return {
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
        "avg_completed_elapsed_ms": round(mean(completed_elapsed_ms), 3) if completed_elapsed_ms else None,
    }


def _build_executive_summary(session_metrics: dict[str, Any], plan_payload: dict[str, Any]) -> str:
    completed_runs = int(session_metrics.get("completed_runs") or 0)
    run_count = int(session_metrics.get("run_count") or 0)
    failed_runs = int(session_metrics.get("failed_runs") or 0)
    model_name = plan_payload.get("model_name")

    if completed_runs == 0:
        return "No completed runs were recorded."

    model_fragment = f" for {model_name}" if model_name else ""
    return f"{completed_runs} of {run_count} runs completed successfully{model_fragment}; {failed_runs} failed."


def _build_model_summaries(run_payloads: list[dict[str, Any]]) -> dict[str, ModelSummary]:
    grouped_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in run_payloads:
        model_name = str(run.get("model_name") or "unknown")
        grouped_runs[model_name].append(run)

    model_summaries: dict[str, ModelSummary] = {}
    for model_name, grouped in grouped_runs.items():
        completed_elapsed_ms = [
            float(run["elapsed_ms"])
            for run in grouped
            if run.get("state") == "completed" and isinstance(run.get("elapsed_ms"), int | float)
        ]
        model_summaries[model_name] = ModelSummary(
            summary=(
                "No completed runs were recorded."
                if not completed_elapsed_ms
                else f"{len(completed_elapsed_ms)} completed run(s) recorded."
            ),
            metrics={
                "run_count": len(grouped),
                "completed_runs": sum(1 for run in grouped if run.get("state") == "completed"),
                "failed_runs": sum(1 for run in grouped if run.get("state") == "failed"),
                "avg_elapsed_ms": round(mean(completed_elapsed_ms), 3) if completed_elapsed_ms else None,
            },
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
        elapsed_values = [
            float(run["elapsed_ms"])
            for run in grouped
            if run.get("state") == "completed" and isinstance(run.get("elapsed_ms"), int | float)
        ]
        tps_values = [
            float(run["metrics"]["tokens_per_second"])
            for run in grouped
            if isinstance(run.get("metrics"), dict)
            and isinstance(run["metrics"].get("tokens_per_second"), int | float)
        ]
        rows.append(
            {
                "model_name": key[0],
                "context_size": key[1],
                "benchmark_type": key[2],
                "scenario_id": key[3],
                "sample_size": len(grouped),
                "completed_runs": sum(1 for run in grouped if run.get("state") == "completed"),
                "failed_runs": sum(1 for run in grouped if run.get("state") == "failed"),
                "stopped_runs": sum(1 for run in grouped if run.get("state") == "stopped"),
                "avg_elapsed_ms": round(mean(elapsed_values), 3) if elapsed_values else None,
                "max_tokens_per_second": round(max(tps_values), 3) if tps_values else None,
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
    if not run_payloads:
        warnings.append("No runs were recorded for this session.")
    return warnings


def _build_recommendations(session_metrics: dict[str, Any]) -> list[str]:
    if int(session_metrics.get("completed_runs") or 0) == 0:
        return ["Run at least one benchmark to generate performance recommendations."]
    return ["Use raw.jsonl for per-run inspection and summary.json for aggregate comparisons."]


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
