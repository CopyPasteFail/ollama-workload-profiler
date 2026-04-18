from __future__ import annotations

from .base import BenchmarkFamily, ExecutionMode
from ..models.plan import BenchmarkType
from ..prompts.scenarios import ScenarioDefinition, TextPromptPayload


def family() -> BenchmarkFamily:
    return BenchmarkFamily(
        benchmark_type=BenchmarkType.SMOKE,
        execution_mode=ExecutionMode.GENERATE,
        scenarios=build_scenarios,
    )


def build_scenarios(_context_size: int) -> list[ScenarioDefinition]:
    return [
        ScenarioDefinition(
            scenario_id="smoke-basic-v1",
            benchmark_type=BenchmarkType.SMOKE,
            name="Smoke sanity check",
            version="v1",
            prompt_payload=TextPromptPayload(
                "Respond with a brief confirmation that the model is ready."
            ),
            target_output_tokens=16,
            difficulty_tag="light",
            phase_emphasis="load_sensitive",
        )
    ]
