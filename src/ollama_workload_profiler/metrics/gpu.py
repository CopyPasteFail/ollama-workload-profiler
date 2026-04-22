from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True, slots=True)
class GpuTelemetrySample:
    available: bool
    source: str | None = None
    memory_used_mb: float | None = None
    util_percent: float | None = None
    device_count: int = 0
    notes: list[str] = field(default_factory=list)
    error: str | None = None


class ExternalGpuTelemetryCollector:
    def __init__(
        self,
        *,
        executable_finder: Callable[[str], str | None] = shutil.which,
        command_runner: Callable[[list[str]], str] | None = None,
    ) -> None:
        self._executable_finder = executable_finder
        self._command_runner = command_runner or _run_command

    def sample(self) -> GpuTelemetrySample:
        nvidia_smi = self._executable_finder("nvidia-smi")
        if nvidia_smi:
            return self._sample_nvidia(nvidia_smi)

        return GpuTelemetrySample(
            available=False,
            notes=[
                "nvidia-smi not found; rocm-smi and Apple Silicon live GPU telemetry are not supported yet"
            ],
        )

    def _sample_nvidia(self, executable: str) -> GpuTelemetrySample:
        try:
            output = self._command_runner(
                [
                    executable,
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]
            )
            readings = _parse_nvidia_smi_output(output)
        except Exception as exc:
            return GpuTelemetrySample(
                available=False,
                source="nvidia-smi",
                error=str(exc) or exc.__class__.__name__,
            )

        if not readings:
            return GpuTelemetrySample(
                available=False,
                source="nvidia-smi",
                error="nvidia-smi returned no parseable GPU telemetry rows",
            )

        memory_values = [reading[0] for reading in readings]
        util_values = [reading[1] for reading in readings]
        notes: list[str] = []
        if len(readings) > 1:
            notes.append("multi_gpu_memory_summed_util_averaged")

        return GpuTelemetrySample(
            available=True,
            source="nvidia-smi",
            memory_used_mb=round(sum(memory_values), 3),
            util_percent=round(sum(util_values) / len(util_values), 3),
            device_count=len(readings),
            notes=notes,
        )


def _run_command(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=1.0,
    )
    return completed.stdout


def _parse_nvidia_smi_output(output: str) -> list[tuple[float, float]]:
    readings: list[tuple[float, float]] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) != 2:
            continue
        try:
            memory_used_mb = float(parts[0])
            util_percent = float(parts[1])
        except ValueError:
            continue
        readings.append((memory_used_mb, util_percent))
    return readings
