from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .sampler import SamplePoint


def compute_phase_peaks(samples: Sequence[SamplePoint | Mapping[str, Any]]) -> dict[str, SamplePoint]:
    peaks: dict[str, SamplePoint] = {}

    for sample in samples:
        point = _coerce_sample_point(sample)
        current = peaks.get(point.phase)
        # Break RSS ties by CPU percent so peak selection is deterministic.
        if current is None or (point.rss_mb, point.cpu_percent) > (current.rss_mb, current.cpu_percent):
            peaks[point.phase] = point

    return peaks


def compute_gpu_telemetry_summary(samples: Sequence[SamplePoint | Mapping[str, Any]]) -> dict[str, Any]:
    points = [_coerce_sample_point(sample) for sample in samples]
    sources = sorted(
        {
            point.gpu_telemetry_source
            for point in points
            if point.gpu_telemetry_source
        }
    )
    notes = sorted(
        {
            note
            for point in points
            for note in point.gpu_telemetry_notes
            if note
        }
    )
    errors = sorted(
        {
            point.gpu_telemetry_error
            for point in points
            if point.gpu_telemetry_error
        }
    )
    available_points = [point for point in points if point.gpu_telemetry_available]
    memory_values = [
        point.gpu_memory_used_mb
        for point in available_points
        if isinstance(point.gpu_memory_used_mb, int | float)
    ]
    util_values = [
        point.gpu_util_percent
        for point in available_points
        if isinstance(point.gpu_util_percent, int | float)
    ]

    return {
        "gpu_telemetry_available": bool(available_points),
        "gpu_telemetry_source": sources[0] if sources else None,
        "gpu_backend": sources[0] if sources else None,
        "peak_gpu_memory_mb": round(max(memory_values), 3) if memory_values else None,
        "avg_gpu_util_percent": round(sum(util_values) / len(util_values), 3) if util_values else None,
        "peak_gpu_util_percent": round(max(util_values), 3) if util_values else None,
        "gpu_device_count": max((point.gpu_device_count for point in points), default=0),
        "gpu_telemetry_notes": notes,
        "gpu_telemetry_errors": errors,
    }


def compute_process_telemetry_summary(samples: Sequence[SamplePoint | Mapping[str, Any]]) -> dict[str, Any]:
    points = [_coerce_sample_point(sample) for sample in samples]
    sampled_process_ids = sorted(
        {
            process_id
            for point in points
            for process_id in point.sampled_process_ids
        }
    )
    return {
        "sampled_process_count": len(sampled_process_ids),
        "sampled_process_ids": sampled_process_ids,
    }


def _coerce_sample_point(sample: SamplePoint | Mapping[str, Any]) -> SamplePoint:
    if isinstance(sample, SamplePoint):
        return sample

    return SamplePoint.from_mapping(dict(sample))
