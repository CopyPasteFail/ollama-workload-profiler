from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence


class PlotExportError(ValueError):
    """Raised when summary artifacts cannot support plot export."""


@dataclass(frozen=True, slots=True)
class PlotPoint:
    x: float
    y: float
    label: str


@dataclass(frozen=True, slots=True)
class PlotSeries:
    name: str
    unit: str
    color: str
    points: tuple[PlotPoint, ...]


LATENCY_SERIES: tuple[tuple[str, str, str, str], ...] = (
    ("TTFT", "ms", "#2563eb", "ttft_ms_median"),
    (
        "Median stream emission interval",
        "ms",
        "#f97316",
        "stream_emission_interval_ms_median_median",
    ),
)
THROUGHPUT_SERIES: tuple[tuple[str, str, str, str], ...] = (
    ("Prompt processing speed", "tokens/s", "#16a34a", "prompt_tokens_per_second_median"),
    ("Generation speed", "tokens/s", "#9333ea", "generation_tokens_per_second_median"),
)


def export_summary_plots(session_dir: Path, *, output_dir: Path | None = None) -> list[Path]:
    """Export lightweight SVG plots from an existing session summary.json."""
    session_path = Path(session_dir)
    summary_path = session_path / "summary.json"
    if not summary_path.exists():
        raise PlotExportError(f"Missing summary artifact: {summary_path}")

    try:
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlotExportError(f"Invalid summary artifact: {summary_path}") from exc

    rows = summary_payload.get("benchmark_summaries")
    if not isinstance(rows, list):
        raise PlotExportError("summary.json does not contain benchmark_summaries.")

    target_dir = Path(output_dir) if output_dir is not None else session_path / "plots"
    target_dir.mkdir(parents=True, exist_ok=True)

    latency_path = target_dir / "latency_vs_prompt_size.svg"
    throughput_path = target_dir / "throughput_vs_prompt_size.svg"
    note_path = target_dir / "README.md"

    latency_path.write_text(
        _render_svg_chart(
            title="TTFT vs prompt size",
            x_label="Prompt size (tokens when available; context size fallback)",
            y_label="Latency (ms)",
            series=_build_series(rows, LATENCY_SERIES),
        ),
        encoding="utf-8",
    )
    throughput_path.write_text(
        _render_svg_chart(
            title="Prompt processing speed vs prompt size",
            x_label="Prompt size (tokens when available; context size fallback)",
            y_label="Throughput (tokens/s)",
            series=_build_series(rows, THROUGHPUT_SERIES),
        ),
        encoding="utf-8",
    )
    note_path.write_text(_render_plot_readme(), encoding="utf-8")
    return [latency_path, throughput_path, note_path]


def _build_series(
    rows: Sequence[Any],
    definitions: Sequence[tuple[str, str, str, str]],
) -> tuple[PlotSeries, ...]:
    series: list[PlotSeries] = []
    for name, unit, color, metric_name in definitions:
        points = [
            point
            for row in rows
            if isinstance(row, Mapping)
            for point in [_plot_point(row, metric_name)]
            if point is not None
        ]
        series.append(
            PlotSeries(
                name=name,
                unit=unit,
                color=color,
                points=tuple(sorted(points, key=lambda point: (point.x, point.label))),
            )
        )
    return tuple(series)


def _plot_point(row: Mapping[str, Any], metric_name: str) -> PlotPoint | None:
    x_value = _prompt_size_value(row)
    y_value = _numeric(row.get(metric_name))
    if x_value is None or y_value is None:
        return None

    label_parts = [
        _optional_text(row.get("benchmark_type")),
        _optional_text(row.get("scenario_id")),
    ]
    return PlotPoint(
        x=x_value,
        y=y_value,
        label=" / ".join(part for part in label_parts if part) or "benchmark row",
    )


def _prompt_size_value(row: Mapping[str, Any]) -> float | None:
    for field_name in (
        "prompt_eval_count_median",
        "actual_prompt_tokens_median",
        "target_prompt_tokens_median",
        "context_size",
    ):
        value = _numeric(row.get(field_name))
        if value is not None:
            return value
    return None


def _render_svg_chart(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[PlotSeries],
) -> str:
    width = 960
    height = 540
    margin_left = 96
    margin_right = 56
    margin_top = 72
    margin_bottom = 88
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    all_points = [point for item in series for point in item.points]

    if not all_points:
        return _render_empty_svg(title=title, message="No compatible summary metrics were available.")

    min_x, max_x = _bounds(point.x for point in all_points)
    min_y, max_y = _bounds((point.y for point in all_points), floor_zero=True)

    def x_coord(value: float) -> float:
        return margin_left + _scale(value, min_x, max_x) * plot_width

    def y_coord(value: float) -> float:
        return margin_top + (1.0 - _scale(value, min_y, max_y)) * plot_height

    elements = [
        _svg_header(width, height),
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" class="title">{escape(title)}</text>',
        f'<text x="{width / 2:.1f}" y="{height - 24}" text-anchor="middle" class="axis-label">{escape(x_label)}</text>',
        (
            f'<text x="26" y="{height / 2:.1f}" text-anchor="middle" class="axis-label" '
            f'transform="rotate(-90 26 {height / 2:.1f})">{escape(y_label)}</text>'
        ),
        _axis_line(margin_left, margin_top + plot_height, margin_left + plot_width, margin_top + plot_height),
        _axis_line(margin_left, margin_top, margin_left, margin_top + plot_height),
    ]
    elements.extend(_tick_elements(min_x, max_x, axis="x", coord=x_coord, baseline=margin_top + plot_height))
    elements.extend(_tick_elements(min_y, max_y, axis="y", coord=y_coord, baseline=margin_left))

    legend_y = 52
    for index, item in enumerate(series):
        legend_x = margin_left + index * 290
        elements.append(f'<circle cx="{legend_x}" cy="{legend_y}" r="5" fill="{item.color}" />')
        elements.append(
            f'<text x="{legend_x + 12}" y="{legend_y + 5}" class="legend">{escape(item.name)} ({escape(item.unit)})</text>'
        )

        if not item.points:
            continue

        point_coords = [(x_coord(point.x), y_coord(point.y), point) for point in item.points]
        if len(point_coords) > 1:
            polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y, _point in point_coords)
            elements.append(
                f'<polyline points="{polyline}" fill="none" stroke="{item.color}" stroke-width="3" />'
            )
        for x, y, point in point_coords:
            elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{item.color}">')
            elements.append(
                f"<title>{escape(point.label)}: x={point.x:g}, {escape(item.name)}={point.y:g} {escape(item.unit)}</title>"
            )
            elements.append("</circle>")

    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def _render_empty_svg(*, title: str, message: str) -> str:
    width = 960
    height = 540
    return "\n".join(
        [
            _svg_header(width, height),
            f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" class="title">{escape(title)}</text>',
            f'<text x="{width / 2:.1f}" y="{height / 2:.1f}" text-anchor="middle" class="empty">{escape(message)}</text>',
            "</svg>",
        ]
    ) + "\n"


def _svg_header(width: int, height: int) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
<style>
  svg {{ background: #fffdf8; color: #172033; font-family: Georgia, 'Times New Roman', serif; }}
  .title {{ font-size: 26px; font-weight: 700; fill: #172033; }}
  .axis-label {{ font-size: 15px; fill: #344054; }}
  .tick {{ font-size: 12px; fill: #475467; }}
  .legend {{ font-size: 14px; fill: #172033; }}
  .empty {{ font-size: 18px; fill: #475467; }}
  .axis {{ stroke: #667085; stroke-width: 1.5; }}
  .grid {{ stroke: #e4e7ec; stroke-width: 1; }}
</style>"""


def _axis_line(x1: float, y1: float, x2: float, y2: float) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="axis" />'


def _tick_elements(
    min_value: float,
    max_value: float,
    *,
    axis: str,
    coord: Any,
    baseline: float,
) -> list[str]:
    values = _tick_values(min_value, max_value)
    elements: list[str] = []
    for value in values:
        if axis == "x":
            x = coord(value)
            elements.append(f'<line x1="{x:.1f}" y1="72" x2="{x:.1f}" y2="{baseline:.1f}" class="grid" />')
            elements.append(f'<text x="{x:.1f}" y="{baseline + 24:.1f}" text-anchor="middle" class="tick">{value:g}</text>')
        else:
            y = coord(value)
            elements.append(f'<line x1="{baseline:.1f}" y1="{y:.1f}" x2="904" y2="{y:.1f}" class="grid" />')
            elements.append(f'<text x="{baseline - 12:.1f}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:g}</text>')
    return elements


def _tick_values(min_value: float, max_value: float) -> list[float]:
    if min_value == max_value:
        return [round(min_value, 3)]
    step = (max_value - min_value) / 4.0
    return [round(min_value + step * index, 3) for index in range(5)]


def _bounds(values: Iterable[float], *, floor_zero: bool = False) -> tuple[float, float]:
    materialized = [float(value) for value in values]
    min_value = min(materialized)
    max_value = max(materialized)
    if floor_zero:
        min_value = min(0.0, min_value)
    if min_value == max_value:
        padding = max(abs(min_value) * 0.1, 1.0)
        return min_value - padding, max_value + padding
    padding = (max_value - min_value) * 0.08
    return min_value - padding, max_value + padding


def _scale(value: float, min_value: float, max_value: float) -> float:
    if min_value == max_value:
        return 0.5
    return (value - min_value) / (max_value - min_value)


def _render_plot_readme() -> str:
    return """# Plot Export Notes

These SVG files are generated from `summary.json` only. They are intended for blog posts and quick hardware comparisons, not as a scientific plotting pipeline.

Prompt size uses the first available field in this order: `prompt_eval_count_median`, `actual_prompt_tokens_median`, `target_prompt_tokens_median`, then `context_size`. If token-count summaries are unavailable, the x-axis may represent context size rather than measured prompt tokens.

`latency_vs_prompt_size.svg` shows TTFT and, when available, median stream-emission interval. Stream cadence is chunk/emission based because local streaming APIs do not provide exact per-token timestamps.

`throughput_vs_prompt_size.svg` shows prompt processing speed and generation speed when those summary metrics are available.
"""


def _numeric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
