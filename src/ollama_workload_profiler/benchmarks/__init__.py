from __future__ import annotations

from . import (
    cold_warm,
    concurrency,
    context_scaling,
    output_scaling,
    profiles,
    prompt_scaling,
    smoke,
    stress,
    ttft,
)
from .base import (
    BenchmarkExecutionStopped,
    BenchmarkFamily,
    BenchmarkRunner,
    ExecutionMode,
    ExecutionRequest,
    ExecutionResult,
)
from ..models.plan import BenchmarkType
from ..prompts.scenarios import ScenarioDefinition

_FAMILIES = {
    BenchmarkType.SMOKE: smoke.family(),
    BenchmarkType.COLD_WARM: cold_warm.family(),
    BenchmarkType.CONCURRENCY_SMOKE: concurrency.family(),
    BenchmarkType.PROMPT_SCALING: prompt_scaling.family(),
    BenchmarkType.CONTEXT_SCALING: context_scaling.family(),
    BenchmarkType.OUTPUT_SCALING: output_scaling.family(),
    BenchmarkType.USE_CASE_PROFILES: profiles.family(),
    BenchmarkType.TTFT: ttft.family(),
    BenchmarkType.STRESS: stress.family(),
}


def resolve_benchmark_family(benchmark_type: BenchmarkType) -> BenchmarkFamily:
    try:
        return _FAMILIES[benchmark_type]
    except KeyError as exc:
        raise NotImplementedError(
            f"Unsupported benchmark type: {benchmark_type.value}"
        ) from exc


def build_scenarios_for_benchmark(
    benchmark_type: BenchmarkType, context_size: int
) -> list[ScenarioDefinition]:
    return resolve_benchmark_family(benchmark_type).resolve_scenarios(context_size)


__all__ = [
    "BenchmarkExecutionStopped",
    "BenchmarkFamily",
    "BenchmarkRunner",
    "ExecutionMode",
    "ExecutionRequest",
    "ExecutionResult",
    "build_scenarios_for_benchmark",
    "resolve_benchmark_family",
]
