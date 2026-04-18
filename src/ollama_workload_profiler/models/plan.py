from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BenchmarkType(StrEnum):
    SMOKE = "smoke"
    COLD_WARM = "cold-warm"
    PROMPT_SCALING = "prompt-scaling"
    CONTEXT_SCALING = "context-scaling"
    OUTPUT_SCALING = "output-scaling"
    USE_CASE_PROFILES = "use-case-profiles"
    TTFT = "ttft"
    STRESS = "stress"


class BenchmarkSessionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    contexts: list[int] = Field(min_length=1)
    benchmark_types: list[BenchmarkType] = Field(min_length=1)
    repetitions: int = Field(default=1, ge=1)
    stop_conditions: list[dict[str, Any]] = Field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None
    execution_settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model_name", mode="before")
    @classmethod
    def _normalize_model_name(cls, value: object) -> object:
        if not isinstance(value, str):
            return value

        model_name = value.strip()
        if not model_name:
            raise ValueError("model_name must not be blank")

        return model_name


class PlannedRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    run_index: int
    model_name: str
    context_size: int
    context_index: int
    benchmark_type: BenchmarkType
    benchmark_type_index: int
    scenario_id: str
    scenario_index: int
    repetition_index: int = Field(ge=1)
    scenario_name: str | None = None
    scenario_version: str | None = None
