from __future__ import annotations

from dataclasses import dataclass

from ..models.plan import BenchmarkType
from .fixtures import (
    MULTI_TURN_CHAT_TURNS,
    PROMPT_SCALING_BASE_TEXT,
    STATIC_CODE_SAMPLE,
    STATIC_SUMMARY_TEXT,
)


@dataclass(frozen=True, slots=True)
class TextPromptPayload:
    text: str


@dataclass(frozen=True, slots=True)
class MultiTurnChatPromptPayload:
    turns: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "turns", tuple(self.turns))


PromptPayload = TextPromptPayload | MultiTurnChatPromptPayload


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    scenario_id: str
    benchmark_type: BenchmarkType
    name: str
    version: str
    prompt_payload: PromptPayload
    target_output_tokens: int
    prompt_template_version: str | None = None
    target_prompt_tokens: int | None = None
    profile_tag: str | None = None
    fill_ratio: float | None = None
    difficulty_tag: str | None = None
    phase_emphasis: str | None = None
    parallelism: int | None = None


def build_scenarios_for_benchmark(
    benchmark_type: BenchmarkType, context_size: int
) -> list[ScenarioDefinition]:
    if benchmark_type is BenchmarkType.PROMPT_SCALING:
        return _build_prompt_scaling_scenarios(benchmark_type)
    if benchmark_type is BenchmarkType.CONTEXT_SCALING:
        return _build_context_scaling_scenarios(benchmark_type, context_size)
    if benchmark_type is BenchmarkType.USE_CASE_PROFILES:
        return _build_use_case_profile_scenarios(benchmark_type)

    raise NotImplementedError(f"Unsupported benchmark type: {benchmark_type.value}")


def _build_prompt_scaling_scenarios(
    benchmark_type: BenchmarkType,
) -> list[ScenarioDefinition]:
    return [
        ScenarioDefinition(
            scenario_id="prompt-scaling-small-v1",
            benchmark_type=benchmark_type,
            name="Small prompt",
            version="v1",
            prompt_payload=TextPromptPayload(
                _repeat_to_length(PROMPT_SCALING_BASE_TEXT, 1024)
            ),
            target_output_tokens=64,
            difficulty_tag="light",
            phase_emphasis="prompt_sensitive",
        ),
        ScenarioDefinition(
            scenario_id="prompt-scaling-medium-v1",
            benchmark_type=benchmark_type,
            name="Medium prompt",
            version="v1",
            prompt_payload=TextPromptPayload(
                _repeat_to_length(PROMPT_SCALING_BASE_TEXT, 4096)
            ),
            target_output_tokens=64,
            difficulty_tag="medium",
            phase_emphasis="prompt_sensitive",
        ),
        ScenarioDefinition(
            scenario_id="prompt-scaling-large-v1",
            benchmark_type=benchmark_type,
            name="Large prompt",
            version="v1",
            prompt_payload=TextPromptPayload(
                _repeat_to_length(PROMPT_SCALING_BASE_TEXT, 16384)
            ),
            target_output_tokens=64,
            difficulty_tag="heavy",
            phase_emphasis="prompt_sensitive",
        ),
    ]


def _build_context_scaling_scenarios(
    benchmark_type: BenchmarkType, context_size: int
) -> list[ScenarioDefinition]:
    return [
        ScenarioDefinition(
            scenario_id="context-scaling-low-v1",
            benchmark_type=benchmark_type,
            name="Low fill context",
            version="v1",
            prompt_payload=TextPromptPayload(PROMPT_SCALING_BASE_TEXT),
            target_output_tokens=64,
            prompt_template_version="v1",
            target_prompt_tokens=max(1, int(context_size * 0.25)),
            fill_ratio=0.25,
            difficulty_tag="light",
            phase_emphasis="load_sensitive",
        ),
        ScenarioDefinition(
            scenario_id="context-scaling-medium-v1",
            benchmark_type=benchmark_type,
            name="Medium fill context",
            version="v1",
            prompt_payload=TextPromptPayload(PROMPT_SCALING_BASE_TEXT),
            target_output_tokens=64,
            prompt_template_version="v1",
            target_prompt_tokens=max(1, int(context_size * 0.5)),
            fill_ratio=0.5,
            difficulty_tag="medium",
            phase_emphasis="load_sensitive",
        ),
        ScenarioDefinition(
            scenario_id="context-scaling-high-v1",
            benchmark_type=benchmark_type,
            name="High fill context",
            version="v1",
            prompt_payload=TextPromptPayload(PROMPT_SCALING_BASE_TEXT),
            target_output_tokens=64,
            prompt_template_version="v1",
            target_prompt_tokens=max(1, int(context_size * 0.8)),
            fill_ratio=0.8,
            difficulty_tag="heavy",
            phase_emphasis="load_sensitive",
        ),
    ]


def _build_use_case_profile_scenarios(
    benchmark_type: BenchmarkType,
) -> list[ScenarioDefinition]:
    return [
        ScenarioDefinition(
            scenario_id="use-case-profiles-quick-qa-v1",
            benchmark_type=benchmark_type,
            name="Quick QA",
            version="v1",
            prompt_payload=TextPromptPayload(
                "Answer the user's question concisely and directly."
            ),
            target_output_tokens=64,
            difficulty_tag="light",
            phase_emphasis="generation_sensitive",
        ),
        ScenarioDefinition(
            scenario_id="use-case-profiles-long-document-summary-v1",
            benchmark_type=benchmark_type,
            name="Long document summary",
            version="v1",
            prompt_payload=TextPromptPayload(STATIC_SUMMARY_TEXT),
            target_output_tokens=128,
            profile_tag="long_document_summary",
            difficulty_tag="medium",
            phase_emphasis="generation_sensitive",
        ),
        ScenarioDefinition(
            scenario_id="use-case-profiles-code-explanation-v1",
            benchmark_type=benchmark_type,
            name="Code explanation",
            version="v1",
            prompt_payload=TextPromptPayload(STATIC_CODE_SAMPLE),
            target_output_tokens=128,
            profile_tag="code_explanation",
            difficulty_tag="medium",
            phase_emphasis="generation_sensitive",
        ),
        ScenarioDefinition(
            scenario_id="use-case-profiles-structured-extraction-v1",
            benchmark_type=benchmark_type,
            name="Structured extraction",
            version="v1",
            prompt_payload=TextPromptPayload(
                "Extract the requested fields into a compact JSON object."
            ),
            target_output_tokens=64,
            profile_tag="structured_extraction",
            difficulty_tag="medium",
            phase_emphasis="generation_sensitive",
        ),
        ScenarioDefinition(
            scenario_id="use-case-profiles-long-form-generation-v1",
            benchmark_type=benchmark_type,
            name="Long form generation",
            version="v1",
            prompt_payload=TextPromptPayload(
                "Write a detailed, multi-paragraph explanation with examples."
            ),
            target_output_tokens=256,
            profile_tag="long_form_generation",
            difficulty_tag="heavy",
            phase_emphasis="generation_sensitive",
        ),
        ScenarioDefinition(
            scenario_id="use-case-profiles-multi-turn-chat-v1",
            benchmark_type=benchmark_type,
            name="Multi-turn chat",
            version="v1",
            prompt_payload=MultiTurnChatPromptPayload(
                turns=MULTI_TURN_CHAT_TURNS,
            ),
            target_output_tokens=96,
            profile_tag="multi_turn_chat",
            difficulty_tag="medium",
            phase_emphasis="generation_sensitive",
        ),
    ]


def _repeat_to_length(text: str, target_length: int) -> str:
    if target_length <= len(text):
        return text[:target_length]

    repeats = (target_length // len(text)) + 1
    combined = text * repeats
    return combined[:target_length]
