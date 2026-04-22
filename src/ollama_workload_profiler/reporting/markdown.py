from __future__ import annotations

from statistics import median

from ..models.verdicts import UseCaseMatrixCell
from ..models.summary import ReportSummary
from .summary import SUMMARY_METRIC_DISPLAY_NAMES, SUMMARY_METRIC_NAMES

MARKDOWN_SUMMARY_METRIC_NAMES: tuple[str, ...] = (
    "elapsed_ms",
    "ttft_ms",
    "prompt_tokens_per_second",
    "generation_tokens_per_second",
    "load_duration_ms",
)

FAMILY_SUMMARY_METRICS: dict[str, tuple[tuple[str, str], ...]] = {
    "smoke": (
        ("elapsed_ms", "elapsed"),
        ("prompt_tokens_per_second", "prompt tokens/s"),
        ("generation_tokens_per_second", "generation tokens/s"),
    ),
    "ttft": (
        ("ttft_ms", "TTFT"),
        ("elapsed_ms", "elapsed"),
    ),
}

FAMILY_DISPLAY_ORDER: tuple[str, ...] = (
    "smoke",
    "ttft",
    "concurrency-smoke",
)


def render_markdown_report(summary: ReportSummary) -> str:
    lines: list[str] = ["# Benchmark Report", ""]
    if summary.benchmark_methodology_version:
        lines.extend([f"Benchmark methodology: {summary.benchmark_methodology_version}", ""])

    lines.extend(["## Executive summary", "", _render_completion_sentence(summary), ""])
    lines.extend(_render_family_summary(summary))

    lines.extend(["## Per-model summary", ""])
    if summary.model_summaries:
        for model_name, model_summary in sorted(summary.model_summaries.items()):
            lines.append(f"### {model_name}")
            lines.append("")
            lines.append(model_summary.summary or "No completed runs were recorded.")
            lines.append("")
            if model_summary.metrics:
                lines.extend(_render_metric_summary_block(model_summary.metrics))
                for metric_name, metric_value in _remaining_metrics(model_summary.metrics):
                    lines.append(f"- `{metric_name}`: {_display_value(metric_value)}")
                lines.append("")
            if model_summary.notes:
                for note in model_summary.notes:
                    lines.append(f"- Note: {note}")
                lines.append("")
    else:
        lines.extend(["No model-level summaries are available.", ""])

    lines.extend(["## Use-case suitability matrix", ""])
    if summary.use_case_matrix:
        lines.extend(_render_table(summary.use_case_matrix))
    else:
        lines.extend(["No use-case profile data is available.", ""])

    lines.extend(["## Detailed timing and resource tables", ""])
    if summary.benchmark_summaries:
        lines.extend(_render_table(summary.benchmark_summaries))
    else:
        lines.extend(["No benchmark aggregates are available.", ""])

    lines.extend(["## Phase-level resource peaks", ""])
    if summary.phase_peak_summaries:
        lines.extend(_render_table(summary.phase_peak_summaries))
    else:
        lines.extend(["No phase peak data is available.", ""])

    lines.extend(["## Warnings and failures", ""])
    if summary.warnings or summary.failures:
        for warning in summary.warnings:
            lines.append(f"- Warning: {warning}")
        for failure in summary.failures:
            lines.append(f"- Failure ({failure.kind}/{failure.phase}): {failure.message}")
        lines.append("")
    else:
        lines.extend(["No warnings or failures were recorded.", ""])

    lines.extend(["## Plain-language recommendations", ""])
    if summary.recommendations:
        for recommendation in summary.recommendations:
            lines.append(f"- {recommendation}")
        lines.append("")
    else:
        lines.extend(["No recommendations are available.", ""])

    return "\n".join(lines).rstrip() + "\n"


def _render_table(rows: list[dict[str, object]]) -> list[str]:
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_display_value(row.get(column)) for column in columns) + " |")
    lines.append("")
    return lines


def _display_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, UseCaseMatrixCell):
        if value.missing_planned_context:
            return "Not measured"
        label = value.verdict_label.display_label if value.verdict_label is not None else "No verdict"
        return f"{label} | {value.supporting_tuple}"
    return str(value)


def _render_completion_sentence(summary: ReportSummary) -> str:
    metrics = summary.session_metrics
    completed_runs = int(metrics.get("completed_runs") or 0)
    run_count = int(metrics.get("run_count") or 0)
    if completed_runs == 0:
        return "No completed runs were recorded."

    model_names = sorted(
        {
            str(row["model_name"])
            for row in summary.benchmark_summaries
            if row.get("model_name") is not None
        }
    )
    model_fragment = f" for {model_names[0]}" if len(model_names) == 1 else ""
    return f"{completed_runs} of {run_count} runs completed successfully{model_fragment}."


def _render_family_summary(summary: ReportSummary) -> list[str]:
    rows_by_family: dict[str, list[dict[str, object]]] = {}
    for row in summary.benchmark_summaries:
        benchmark_type = row.get("benchmark_type")
        if benchmark_type is not None:
            rows_by_family.setdefault(str(benchmark_type), []).append(row)

    family_lines: list[str] = []
    for benchmark_type in _ordered_families(rows_by_family):
        rows = rows_by_family[benchmark_type]
        if benchmark_type == "concurrency-smoke":
            line = _render_concurrency_family_summary(rows)
        else:
            line = _render_standard_family_summary(benchmark_type, rows)
        if line:
            family_lines.append(line)

    if not family_lines:
        return []
    return ["By benchmark family:", *family_lines, ""]


def _ordered_families(rows_by_family: dict[str, list[dict[str, object]]]) -> list[str]:
    ordered = [family for family in FAMILY_DISPLAY_ORDER if family in rows_by_family]
    ordered.extend(sorted(family for family in rows_by_family if family not in FAMILY_DISPLAY_ORDER))
    return ordered


def _render_standard_family_summary(benchmark_type: str, rows: list[dict[str, object]]) -> str | None:
    metric_specs = FAMILY_SUMMARY_METRICS.get(benchmark_type, (("elapsed_ms", "elapsed"),))
    parts: list[str] = []
    for metric_name, label in metric_specs:
        value = _median_summary_value(rows, f"{metric_name}_median")
        if value is not None:
            parts.append(f"{label} median {_display_value(value)}")
    if not parts:
        return None
    return f"- {_family_display_name(benchmark_type)}: {'; '.join(parts)}."


def _render_concurrency_family_summary(rows: list[dict[str, object]]) -> str | None:
    parts: list[str] = []
    for row in sorted(rows, key=_concurrency_row_sort_key):
        parallelism = _concurrency_parallelism(row)
        elapsed = row.get("elapsed_ms_median")
        if parallelism is not None and elapsed is not None:
            parts.append(f"p={parallelism} elapsed median {_display_value(elapsed)}")

    request_ttft_p50 = _median_summary_value(rows, "concurrency_request_ttft_ms_p50_median")
    if request_ttft_p50 is not None:
        parts.append(f"request TTFT p50 median {_display_value(request_ttft_p50)}")

    if not parts:
        return None
    return f"- Concurrency-smoke: {'; '.join(parts)}."


def _median_summary_value(rows: list[dict[str, object]], key: str) -> object | None:
    values = [
        float(value)
        for row in rows
        for value in [row.get(key)]
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    if not values:
        return None
    return round(float(median(values)), 3)


def _concurrency_row_sort_key(row: dict[str, object]) -> tuple[int, str]:
    parallelism = _concurrency_parallelism(row)
    return (parallelism if parallelism is not None else 0, str(row.get("scenario_id") or ""))


def _concurrency_parallelism(row: dict[str, object]) -> int | None:
    scenario_id = row.get("scenario_id")
    if not isinstance(scenario_id, str):
        return None
    marker_index = scenario_id.find("-p")
    if marker_index < 0:
        return None
    suffix = scenario_id[marker_index + 2 :]
    digits: list[str] = []
    for character in suffix:
        if not character.isdigit():
            break
        digits.append(character)
    return int("".join(digits)) if digits else None


def _family_display_name(benchmark_type: str) -> str:
    if benchmark_type == "ttft":
        return "TTFT"
    return benchmark_type[:1].upper() + benchmark_type[1:]


def _render_metric_summary_block(metrics: dict[str, object]) -> list[str]:
    lines: list[str] = []
    for metric_name in MARKDOWN_SUMMARY_METRIC_NAMES:
        sample_size = metrics.get(f"{metric_name}_sample_size")
        median_value = metrics.get(f"{metric_name}_median")
        p95_value = metrics.get(f"{metric_name}_p95")
        display_name = _metric_display_name(metric_name)
        if sample_size:
            lines.append(
                f"- {display_name}: median {_display_value(median_value)} | "
                f"p95 {_display_value(p95_value)} | n={sample_size}"
            )
        else:
            lines.append(f"- {display_name}: unavailable | n=0")
    return lines


def _metric_display_name(metric_name: str) -> str:
    display_name = SUMMARY_METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    return display_name[:1].upper() + display_name[1:]


def _remaining_metrics(metrics: dict[str, object]) -> list[tuple[str, object]]:
    hidden_keys = {"tokens_per_second"}
    for metric_name in SUMMARY_METRIC_NAMES:
        hidden_keys.add(f"{metric_name}_sample_size")
        hidden_keys.add(f"{metric_name}_median")
        hidden_keys.add(f"{metric_name}_p95")

    return sorted(
        (
            (metric_name, metric_value)
            for metric_name, metric_value in metrics.items()
            if metric_name not in hidden_keys
        ),
        key=lambda item: item[0],
    )
