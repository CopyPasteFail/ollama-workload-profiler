from __future__ import annotations

from ..models.verdicts import UseCaseMatrixCell
from ..models.summary import ReportSummary


def render_markdown_report(summary: ReportSummary) -> str:
    lines: list[str] = ["# Benchmark Report", ""]
    if summary.benchmark_methodology_version:
        lines.extend([f"Benchmark methodology: {summary.benchmark_methodology_version}", ""])

    lines.extend(["## Executive summary", "", summary.executive_summary or "No summary available.", ""])
    lines.extend(_render_session_summary(summary))

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


def _render_session_summary(summary: ReportSummary) -> list[str]:
    metrics = summary.session_metrics
    if not metrics:
        return []

    lines = [
        f"Completed samples: {metrics.get('completed_sample_size', 0)} of {metrics.get('run_count', 0)} runs",
        "",
    ]
    lines.extend(_render_metric_summary_block(metrics))
    if len(lines) > 2:
        lines.append("")
    return lines


def _render_metric_summary_block(metrics: dict[str, object]) -> list[str]:
    lines: list[str] = []
    for metric_name in (
        "elapsed_ms",
        "ttft_ms",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
        "load_duration_ms",
    ):
        sample_size = metrics.get(f"{metric_name}_sample_size")
        median_value = metrics.get(f"{metric_name}_median")
        p95_value = metrics.get(f"{metric_name}_p95")
        if sample_size:
            lines.append(f"- {metric_name}: median {_display_value(median_value)} | p95 {_display_value(p95_value)} | n={sample_size}")
        else:
            lines.append(f"- {metric_name}: unavailable | n=0")
    return lines


def _remaining_metrics(metrics: dict[str, object]) -> list[tuple[str, object]]:
    hidden_keys = {"tokens_per_second"}
    for metric_name in (
        "elapsed_ms",
        "ttft_ms",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
        "load_duration_ms",
    ):
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
