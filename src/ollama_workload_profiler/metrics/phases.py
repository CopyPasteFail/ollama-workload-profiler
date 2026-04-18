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


def _coerce_sample_point(sample: SamplePoint | Mapping[str, Any]) -> SamplePoint:
    if isinstance(sample, SamplePoint):
        return sample

    return SamplePoint.from_mapping(dict(sample))
