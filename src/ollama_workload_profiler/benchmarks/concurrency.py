from __future__ import annotations

from .base import BenchmarkFamily, ExecutionMode
from ..models.plan import BenchmarkType
from ..prompts.scenarios import ScenarioDefinition, TextPromptPayload


def family() -> BenchmarkFamily:
    return BenchmarkFamily(
        benchmark_type=BenchmarkType.CONCURRENCY_SMOKE,
        execution_mode=ExecutionMode.CONCURRENCY,
        scenarios=build_scenarios,
    )


def build_scenarios(_context_size: int) -> list[ScenarioDefinition]:
    return [
        _scenario(parallelism=2),
        _scenario(parallelism=4),
    ]


def _scenario(*, parallelism: int) -> ScenarioDefinition:
    return ScenarioDefinition(
        scenario_id=f"concurrency-smoke-p{parallelism}-v1",
        benchmark_type=BenchmarkType.CONCURRENCY_SMOKE,
        name=f"Concurrency smoke p={parallelism}",
        version="v1",
        prompt_payload=TextPromptPayload(
            "Briefly confirm readiness while another local request may be running."
        ),
        target_output_tokens=32,
        difficulty_tag="light",
        phase_emphasis="contention_sensitive",
        parallelism=parallelism,
    )
