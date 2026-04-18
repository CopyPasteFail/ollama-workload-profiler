from __future__ import annotations

from .base import BenchmarkFamily, ExecutionMode
from ..models.plan import BenchmarkType
from ..prompts.scenarios import build_scenarios_for_benchmark


def family() -> BenchmarkFamily:
    return BenchmarkFamily(
        benchmark_type=BenchmarkType.USE_CASE_PROFILES,
        execution_mode=ExecutionMode.GENERATE,
        scenarios=build_scenarios,
    )


def build_scenarios(context_size: int):
    return build_scenarios_for_benchmark(BenchmarkType.USE_CASE_PROFILES, context_size)
