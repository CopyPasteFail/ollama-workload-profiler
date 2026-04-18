from __future__ import annotations

from .base import BenchmarkFamily, ExecutionMode
from ..models.plan import BenchmarkType
from ..prompts.scenarios import ScenarioDefinition, TextPromptPayload


def family() -> BenchmarkFamily:
    return BenchmarkFamily(
        benchmark_type=BenchmarkType.OUTPUT_SCALING,
        execution_mode=ExecutionMode.GENERATE,
        scenarios=build_scenarios,
    )


def build_scenarios(_context_size: int) -> list[ScenarioDefinition]:
    return [
        ScenarioDefinition(
            scenario_id="output-scaling-short-v1",
            benchmark_type=BenchmarkType.OUTPUT_SCALING,
            name="Short output",
            version="v1",
            prompt_payload=TextPromptPayload("Answer in one short sentence."),
            target_output_tokens=32,
            difficulty_tag="light",
            phase_emphasis="generation_sensitive",
        ),
        ScenarioDefinition(
            scenario_id="output-scaling-medium-v1",
            benchmark_type=BenchmarkType.OUTPUT_SCALING,
            name="Medium output",
            version="v1",
            prompt_payload=TextPromptPayload("Answer in two concise paragraphs."),
            target_output_tokens=128,
            difficulty_tag="medium",
            phase_emphasis="generation_sensitive",
        ),
        ScenarioDefinition(
            scenario_id="output-scaling-long-v1",
            benchmark_type=BenchmarkType.OUTPUT_SCALING,
            name="Long output",
            version="v1",
            prompt_payload=TextPromptPayload("Answer in a detailed multi-section response."),
            target_output_tokens=768,
            difficulty_tag="heavy",
            phase_emphasis="generation_sensitive",
        ),
    ]
