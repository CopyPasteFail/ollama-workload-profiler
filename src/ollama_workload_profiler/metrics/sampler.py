from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from collections.abc import Callable
from typing import Any

from .process import find_ollama_processes


@dataclass(frozen=True, slots=True)
class SamplePoint:
    phase: str
    rss_mb: float
    cpu_percent: float

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
        )


class PollingProcessSampler:
    def __init__(
        self,
        *,
        process_finder: Callable[[], list[Any]] = find_ollama_processes,
        interval_seconds: float = 0.01,
        shutdown_timeout_seconds: float = 0.1,
    ) -> None:
        self._process_finder = process_finder
        self._interval_seconds = interval_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[SamplePoint] = []
        self._samples_lock = threading.Lock()
        self._discard_future_samples = False

    def start(self, run: object) -> None:
        del run
        self._samples = []
        self._stop_event.clear()
        self._discard_future_samples = False
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
        for process in self._process_finder():
            try:
                memory_info = process.memory_info()
                rss_total_mb += float(memory_info.rss) / (1024 * 1024)
                cpu_total += float(process.cpu_percent(interval=None))
            except Exception:
                continue

        return SamplePoint(
            phase=phase,
            rss_mb=round(rss_total_mb, 3),
            cpu_percent=round(cpu_total, 3),
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
