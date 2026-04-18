from __future__ import annotations

from .base import BenchmarkFamily, ExecutionMode
from ..models.plan import BenchmarkType
from ..prompts.scenarios import ScenarioDefinition, TextPromptPayload


def family() -> BenchmarkFamily:
    return BenchmarkFamily(
        benchmark_type=BenchmarkType.COLD_WARM,
        execution_mode=ExecutionMode.GENERATE,
        scenarios=build_scenarios,
    )


def build_scenarios(_context_size: int) -> list[ScenarioDefinition]:
    return [
        ScenarioDefinition(
            scenario_id="cold-warm-cold-start-v1",
            benchmark_type=BenchmarkType.COLD_WARM,
            name="Cold start probe",
            version="v1",
            prompt_payload=TextPromptPayload("Respond with a cold-start acknowledgement."),
            target_output_tokens=16,
            profile_tag="cold_start",
            difficulty_tag="medium",
            phase_emphasis="load_sensitive",
        ),
        ScenarioDefinition(
            scenario_id="cold-warm-warm-start-v1",
            benchmark_type=BenchmarkType.COLD_WARM,
            name="Warm start probe",
            version="v1",
            prompt_payload=TextPromptPayload("Respond with a warm-start acknowledgement."),
            target_output_tokens=16,
            profile_tag="warm_start",
            difficulty_tag="light",
            phase_emphasis="load_sensitive",
        ),
    ]
