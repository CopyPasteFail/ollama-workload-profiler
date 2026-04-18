from __future__ import annotations

from .base import BenchmarkFamily, ExecutionMode
from ..models.plan import BenchmarkType
from ..prompts.scenarios import (
    MultiTurnChatPromptPayload,
    ScenarioDefinition,
    TextPromptPayload,
)


def family() -> BenchmarkFamily:
    return BenchmarkFamily(
        benchmark_type=BenchmarkType.TTFT,
        execution_mode=ExecutionMode.TTFT,
        scenarios=build_scenarios,
    )


def build_scenarios(_context_size: int) -> list[ScenarioDefinition]:
    return [
        ScenarioDefinition(
            scenario_id="ttft-basic-v1",
            benchmark_type=BenchmarkType.TTFT,
            name="TTFT probe",
            version="v1",
            prompt_payload=TextPromptPayload("Reply with a single token."),
            target_output_tokens=1,
            difficulty_tag="light",
            phase_emphasis="load_sensitive",
        ),
        ScenarioDefinition(
            scenario_id="ttft-chat-v1",
            benchmark_type=BenchmarkType.TTFT,
            name="TTFT chat probe",
            version="v1",
            prompt_payload=MultiTurnChatPromptPayload(
                (
                    "Reply with a single token.",
                    "Reply again with the same single token.",
                )
            ),
            target_output_tokens=1,
            difficulty_tag="light",
            phase_emphasis="load_sensitive",
        ),
    ]
