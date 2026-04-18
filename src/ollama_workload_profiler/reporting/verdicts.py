from __future__ import annotations

from statistics import mean
from typing import Any, Mapping, Sequence

from ..models.results import RunState
from ..models.verdicts import UseCaseMatrixCell, Verdict, VerdictLabel

_TTFT_GOOD_MS = 1500.0
_TTFT_CAUTION_MS = 4000.0
_TTFT_SHORT_TASK_MS = 8000.0

_TPS_GOOD = 20.0
_TPS_CAUTION = 10.0
_TPS_SHORT_TASK = 5.0

_RAM_GOOD_GB = 2.0
_RAM_CAUTION_GB = 1.0
_RAM_SHORT_TASK_GB = 0.5

_FAILURE_RATE_GOOD = 0.0
_FAILURE_RATE_CAUTION = 0.34
_FAILURE_RATE_SHORT_TASK = 0.67


def classify_verdict(
    *,
    success: bool,
    ttft_ms: float | None = None,
    gen_tokens_per_second: float | None = None,
    ram_headroom_gb: float | None = None,
    failure_rate: float | None = None,
    sample_count: int | None = None,
) -> Verdict:
    supporting_metrics: dict[str, str | int | float | bool | None] = {}
    if ttft_ms is not None:
        supporting_metrics["ttft_ms"] = _round_metric(ttft_ms)
    if gen_tokens_per_second is not None:
        supporting_metrics["gen_tokens_per_second"] = _round_metric(gen_tokens_per_second)
    if ram_headroom_gb is not None:
        supporting_metrics["ram_headroom_gb"] = _round_metric(ram_headroom_gb)
    if failure_rate is not None:
        supporting_metrics["failure_rate"] = _round_metric(failure_rate)
    if sample_count is not None:
        supporting_metrics["sample_count"] = sample_count

    if not success:
        return Verdict(
            label=VerdictLabel.AVOID_ON_THIS_HARDWARE,
            rationale="Run did not complete successfully.",
            supporting_metrics=supporting_metrics,
        )

    severity = 0
    available_metrics = 0
    rationale_parts: list[str] = []

    if ttft_ms is not None:
        available_metrics += 1
        severity = max(severity, _severity_from_lower_is_better(ttft_ms, _TTFT_GOOD_MS, _TTFT_CAUTION_MS, _TTFT_SHORT_TASK_MS))
        rationale_parts.append(f"ttft_ms={_round_metric(ttft_ms)}")
    if gen_tokens_per_second is not None:
        available_metrics += 1
        severity = max(severity, _severity_from_higher_is_better(gen_tokens_per_second, _TPS_GOOD, _TPS_CAUTION, _TPS_SHORT_TASK))
        rationale_parts.append(f"gen_tokens_per_second={_round_metric(gen_tokens_per_second)}")
    if ram_headroom_gb is not None:
        available_metrics += 1
        severity = max(severity, _severity_from_higher_is_better(ram_headroom_gb, _RAM_GOOD_GB, _RAM_CAUTION_GB, _RAM_SHORT_TASK_GB))
        rationale_parts.append(f"ram_headroom_gb={_round_metric(ram_headroom_gb)}")
    if failure_rate is not None:
        severity = max(severity, _severity_from_failure_rate(failure_rate))
        rationale_parts.append(f"failure_rate={_round_metric(failure_rate)}")

    if available_metrics == 0:
        severity = max(severity, 2)
        rationale = "No performance metrics were available."
    elif available_metrics == 1 and severity == 0:
        severity = 1
        rationale = "Only one metric was available, so confidence is limited."
    else:
        rationale = _rationale_from_severity(severity)

    if rationale_parts:
        rationale = f"{rationale} Signals: {', '.join(rationale_parts)}."

    return Verdict(
        label=_severity_to_label(severity),
        rationale=rationale,
        supporting_metrics=supporting_metrics,
    )


def summarize_use_case_cell(
    *,
    runs: Sequence[Mapping[str, Any]],
) -> UseCaseMatrixCell:
    total_runs = len(runs)
    if total_runs == 0:
        return build_missing_use_case_cell()

    completed_runs = [run for run in runs if _run_state_value(run) == RunState.COMPLETED.value]
    failed_runs = [run for run in runs if _run_state_value(run) == RunState.FAILED.value]
    stopped_runs = [run for run in runs if _run_state_value(run) == RunState.STOPPED.value]

    elapsed_ms_values = [
        float(run["elapsed_ms"])
        for run in completed_runs
        if isinstance(run.get("elapsed_ms"), int | float)
    ]
    tps_values = [
        float(metrics["tokens_per_second"])
        for run in completed_runs
        if isinstance((metrics := run.get("metrics")), dict)
        and isinstance(metrics.get("tokens_per_second"), int | float)
    ]
    ttft_values = [
        float(metrics["ttft_ms"])
        for run in completed_runs
        if isinstance((metrics := run.get("metrics")), dict)
        and isinstance(metrics.get("ttft_ms"), int | float)
    ]
    ram_headroom_values = [
        float(metrics["ram_headroom_gb"])
        for run in completed_runs
        if isinstance((metrics := run.get("metrics")), dict)
        and isinstance(metrics.get("ram_headroom_gb"), int | float)
    ]

    completed_count = len(completed_runs)
    failure_rate = (len(failed_runs) + len(stopped_runs)) / total_runs if total_runs else None

    verdict = classify_verdict(
        success=completed_count > 0,
        ttft_ms=mean(ttft_values) if ttft_values else None,
        gen_tokens_per_second=mean(tps_values) if tps_values else None,
        ram_headroom_gb=mean(ram_headroom_values) if ram_headroom_values else None,
        failure_rate=failure_rate,
        sample_count=completed_count,
    )

    return UseCaseMatrixCell(
        verdict_label=verdict.label,
        supporting_tuple=_build_supporting_tuple(
            planned_runs=total_runs,
            total_runs=total_runs,
            completed_runs=completed_count,
            failed_runs=len(failed_runs),
            stopped_runs=len(stopped_runs),
            avg_elapsed_ms=_round_metric(mean(elapsed_ms_values)) if elapsed_ms_values else None,
            avg_tokens_per_second=_round_metric(mean(tps_values)) if tps_values else None,
            avg_ttft_ms=_round_metric(mean(ttft_values)) if ttft_values else None,
            avg_ram_headroom_gb=_round_metric(mean(ram_headroom_values)) if ram_headroom_values else None,
            failure_rate=_round_metric(failure_rate) if failure_rate is not None else None,
        ),
    )


def build_missing_use_case_cell() -> UseCaseMatrixCell:
    return UseCaseMatrixCell(
        verdict_label=None,
        supporting_tuple=_build_supporting_tuple(
            planned_runs=0,
            total_runs=0,
            completed_runs=0,
            failed_runs=0,
            stopped_runs=0,
            avg_elapsed_ms=None,
            avg_tokens_per_second=None,
            avg_ttft_ms=None,
            avg_ram_headroom_gb=None,
            failure_rate=None,
        ),
        missing_planned_context=True,
    )


def _build_supporting_tuple(
    *,
    planned_runs: int,
    total_runs: int,
    completed_runs: int,
    failed_runs: int,
    stopped_runs: int,
    avg_elapsed_ms: float | None,
    avg_tokens_per_second: float | None,
    avg_ttft_ms: float | None,
    avg_ram_headroom_gb: float | None,
    failure_rate: float | None,
) -> tuple[int, int, int, int, int, float | None, float | None, float | None, float | None, float | None]:
    return (
        planned_runs,
        total_runs,
        completed_runs,
        failed_runs,
        stopped_runs,
        avg_elapsed_ms,
        avg_tokens_per_second,
        avg_ttft_ms,
        avg_ram_headroom_gb,
        failure_rate,
    )


def _severity_from_lower_is_better(value: float, good: float, caution: float, short_task: float) -> int:
    if value <= good:
        return 0
    if value <= caution:
        return 1
    if value <= short_task:
        return 3
    return 4


def _severity_from_higher_is_better(value: float, good: float, caution: float, short_task: float) -> int:
    if value >= good:
        return 0
    if value >= caution:
        return 1
    if value >= short_task:
        return 3
    return 4


def _severity_from_failure_rate(value: float) -> int:
    if value <= _FAILURE_RATE_GOOD:
        return 0
    if value <= _FAILURE_RATE_CAUTION:
        return 1
    if value <= _FAILURE_RATE_SHORT_TASK:
        return 3
    return 4


def _severity_to_label(severity: int) -> VerdictLabel:
    if severity <= 0:
        return VerdictLabel.GOOD_FIT
    if severity == 1:
        return VerdictLabel.GOOD_FIT_WITH_CAUTION
    if severity == 2:
        return VerdictLabel.USE_WITH_CAUTION
    if severity == 3:
        return VerdictLabel.USE_ONLY_FOR_SHORT_TASKS
    return VerdictLabel.AVOID_ON_THIS_HARDWARE


def _rationale_from_severity(severity: int) -> str:
    if severity <= 0:
        return "Observed metrics are comfortably within the preferred bands."
    if severity == 1:
        return "Observed metrics are usable, but confidence is limited."
    if severity == 2:
        return "Observed metrics are incomplete or borderline."
    if severity == 3:
        return "Observed metrics suggest this hardware is best kept to short tasks."
    return "Observed metrics indicate this hardware is a poor fit."


def _round_metric(value: float) -> float:
    return round(float(value), 3)


def _run_state_value(run: Mapping[str, Any]) -> str:
    state = run.get("state")
    return state.value if isinstance(state, RunState) else str(state)
