from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
from collections.abc import Callable
from time import perf_counter
from typing import Any

from .gpu import ExternalGpuTelemetryCollector, GpuTelemetrySample
from .process import find_ollama_processes


@dataclass(frozen=True, slots=True)
class SamplePoint:
    phase: str
    rss_mb: float
    cpu_percent: float
    sampled_process_count: int = 0
    sampled_process_ids: list[int] = field(default_factory=list)
    gpu_telemetry_available: bool = False
    gpu_telemetry_source: str | None = None
    gpu_memory_used_mb: float | None = None
    gpu_util_percent: float | None = None
    gpu_device_count: int = 0
    gpu_telemetry_notes: list[str] = field(default_factory=list)
    gpu_telemetry_error: str | None = None

    @classmethod
    def from_mapping(cls, sample: dict[str, Any]) -> "SamplePoint":
        phase = sample.get("phase")
        rss_mb = sample.get("rss_mb")
        cpu_percent = sample.get("cpu_percent")

        if not isinstance(phase, str) or isinstance(phase, bool) or not phase:
            raise ValueError("sample.phase must be a non-empty string")
        if isinstance(rss_mb, bool) or not isinstance(rss_mb, (int, float)):
            raise ValueError("sample.rss_mb must be a finite number")
        if isinstance(cpu_percent, bool) or not isinstance(cpu_percent, (int, float)):
            raise ValueError("sample.cpu_percent must be a finite number")

        rss_value = float(rss_mb)
        cpu_value = float(cpu_percent)
        if not math.isfinite(rss_value):
            raise ValueError("sample.rss_mb must be finite")
        if not math.isfinite(cpu_value):
            raise ValueError("sample.cpu_percent must be finite")

        return cls(
            phase=phase,
            rss_mb=rss_value,
            cpu_percent=cpu_value,
            sampled_process_count=_optional_nonnegative_int(sample.get("sampled_process_count")),
            sampled_process_ids=_optional_int_list(sample.get("sampled_process_ids")),
            gpu_telemetry_available=bool(sample.get("gpu_telemetry_available", False)),
            gpu_telemetry_source=_optional_string(sample.get("gpu_telemetry_source")),
            gpu_memory_used_mb=_optional_finite_number(sample.get("gpu_memory_used_mb")),
            gpu_util_percent=_optional_finite_number(sample.get("gpu_util_percent")),
            gpu_device_count=_optional_nonnegative_int(sample.get("gpu_device_count")),
            gpu_telemetry_notes=_optional_string_list(sample.get("gpu_telemetry_notes")),
            gpu_telemetry_error=_optional_string(sample.get("gpu_telemetry_error")),
        )


class PollingProcessSampler:
    def __init__(
        self,
        *,
        process_finder: Callable[[], list[Any]] = find_ollama_processes,
        gpu_collector: Any | None = None,
        interval_seconds: float = 0.01,
        gpu_poll_interval_seconds: float = 0.5,
        shutdown_timeout_seconds: float = 0.1,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._process_finder = process_finder
        self._gpu_collector = gpu_collector if gpu_collector is not None else ExternalGpuTelemetryCollector()
        self._interval_seconds = interval_seconds
        self._gpu_poll_interval_seconds = gpu_poll_interval_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._clock = clock
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[SamplePoint] = []
        self._samples_lock = threading.Lock()
        self._discard_future_samples = False
        self._gpu_sample_lock = threading.Lock()
        self._last_gpu_sample: GpuTelemetrySample | None = None
        self._last_gpu_sample_at: float | None = None
        self._gpu_backend_missing_for_run = False

    def start(self, run: object) -> None:
        del run
        self._samples = []
        self._stop_event.clear()
        self._discard_future_samples = False
        self._reset_gpu_cache()
        self._record("load")
        self._thread = threading.Thread(target=self._poll_generation_phase, daemon=True)
        self._thread.start()

    def stop(self) -> list[SamplePoint]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._shutdown_timeout_seconds)
            self._thread = None
        self._discard_future_samples = True
        post_run_sample = self._collect_sample_with_timeout(
            "resident_post_run",
            timeout_seconds=self._shutdown_timeout_seconds,
        )
        if post_run_sample is not None:
            self._append_sample(post_run_sample, allow_after_stop=True)
        with self._samples_lock:
            return list(self._samples)

    def _poll_generation_phase(self) -> None:
        self._record("generation")
        while not self._stop_event.wait(self._interval_seconds):
            self._record("generation")

    def _record(self, phase: str) -> None:
        self._append_sample(self._snapshot_sample(phase))

    def _snapshot_sample(self, phase: str) -> SamplePoint:
        rss_total_mb = 0.0
        cpu_total = 0.0
        sampled_process_ids: list[int] = []
        for process in self._process_finder():
            try:
                memory_info = process.memory_info()
                rss_total_mb += float(memory_info.rss) / (1024 * 1024)
                cpu_total += float(process.cpu_percent(interval=None))
                process_id = _process_id(process)
                if process_id is not None:
                    sampled_process_ids.append(process_id)
            except Exception:
                continue

        sampled_process_ids = sorted(set(sampled_process_ids))
        gpu_sample = self._cached_gpu_sample()
        return SamplePoint(
            phase=phase,
            rss_mb=round(rss_total_mb, 3),
            cpu_percent=round(cpu_total, 3),
            sampled_process_count=len(sampled_process_ids),
            sampled_process_ids=sampled_process_ids,
            gpu_telemetry_available=gpu_sample.available,
            gpu_telemetry_source=gpu_sample.source,
            gpu_memory_used_mb=gpu_sample.memory_used_mb,
            gpu_util_percent=gpu_sample.util_percent,
            gpu_device_count=gpu_sample.device_count,
            gpu_telemetry_notes=list(gpu_sample.notes),
            gpu_telemetry_error=gpu_sample.error,
        )

    def _append_sample(self, sample: SamplePoint, *, allow_after_stop: bool = False) -> None:
        with self._samples_lock:
            if self._discard_future_samples and not allow_after_stop:
                return
            self._samples.append(sample)

    def _collect_sample_with_timeout(
        self,
        phase: str,
        *,
        timeout_seconds: float,
    ) -> SamplePoint | None:
        collected: list[SamplePoint] = []

        def worker() -> None:
            collected.append(self._snapshot_sample(phase))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)
        if thread.is_alive() or not collected:
            return None
        return collected[0]

    def _safe_gpu_sample(self) -> GpuTelemetrySample:
        try:
            return self._gpu_collector.sample()
        except Exception as exc:
            return GpuTelemetrySample(
                available=False,
                error=str(exc) or exc.__class__.__name__,
            )

    def _cached_gpu_sample(self) -> GpuTelemetrySample:
        now = float(self._clock())
        with self._gpu_sample_lock:
            if self._last_gpu_sample is not None:
                if self._gpu_backend_missing_for_run:
                    return self._last_gpu_sample
                last_sample_at = self._last_gpu_sample_at
                if last_sample_at is not None and now - last_sample_at < self._gpu_poll_interval_seconds:
                    return self._last_gpu_sample

            sample = self._safe_gpu_sample()
            self._last_gpu_sample = sample
            self._last_gpu_sample_at = now
            if _is_missing_gpu_backend_sample(sample):
                self._gpu_backend_missing_for_run = True
            return sample

    def _reset_gpu_cache(self) -> None:
        with self._gpu_sample_lock:
            self._last_gpu_sample = None
            self._last_gpu_sample_at = None
            self._gpu_backend_missing_for_run = False


def _optional_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _optional_nonnegative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _optional_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value or None


def _optional_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _optional_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return sorted({item for item in value if isinstance(item, int) and not isinstance(item, bool)})


def _is_missing_gpu_backend_sample(sample: Any) -> bool:
    return (
        getattr(sample, "available", False) is False
        and getattr(sample, "source", None) is None
        and getattr(sample, "error", None) is None
    )


def _process_id(process: Any) -> int | None:
    info = getattr(process, "info", None)
    if isinstance(info, dict):
        value = info.get("pid")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    value = getattr(process, "pid", None)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None
