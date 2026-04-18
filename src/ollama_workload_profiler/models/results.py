from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .failures import FailureInfo
from .plan import BenchmarkType


class RunState(StrEnum):
    PLANNED = "planned"
    STARTING = "starting"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    run_index: int
    model_name: str
    context_size: int
    context_index: int = Field(ge=1)
    benchmark_type: BenchmarkType
    benchmark_type_index: int = Field(ge=1)
    scenario_id: str
    scenario_index: int = Field(ge=1)
    scenario_version: str | None = None
    state: RunState
    repetition_index: int = Field(default=1, ge=1)
    scenario_name: str | None = None
    elapsed_ms: float | None = None
    partial_result: bool = False
    failure: FailureInfo | None = None
    system_snapshot: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
