from __future__ import annotations

from .base import BenchmarkFamily, ExecutionMode
from ..models.plan import BenchmarkType
from ..prompts.scenarios import ScenarioDefinition, TextPromptPayload


def family() -> BenchmarkFamily:
    return BenchmarkFamily(
        benchmark_type=BenchmarkType.STRESS,
        execution_mode=ExecutionMode.GENERATE,
        scenarios=build_scenarios,
    )


def build_scenarios(context_size: int) -> list[ScenarioDefinition]:
    return [
        ScenarioDefinition(
            scenario_id="stress-sustained-load-v1",
            benchmark_type=BenchmarkType.STRESS,
            name="Sustained load",
            version="v1",
            prompt_payload=TextPromptPayload(
                "Sustain repeated generation with a context footprint sized for stress "
                f"validation around {context_size} tokens."
            ),
            target_output_tokens=256,
            difficulty_tag="heavy",
            phase_emphasis="generation_sensitive",
        ),
        ScenarioDefinition(
            scenario_id="stress-burst-load-v1",
            benchmark_type=BenchmarkType.STRESS,
            name="Burst load",
            version="v1",
            prompt_payload=TextPromptPayload(
                "Produce a bursty, high-throughput response under a similarly large "
                f"stress context around {context_size} tokens."
            ),
            target_output_tokens=384,
            difficulty_tag="heavy",
            phase_emphasis="generation_sensitive",
        ),
    ]
