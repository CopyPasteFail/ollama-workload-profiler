from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FailureInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = "unknown"
    phase: str = "unknown"
    message: str
    exception_class: str | None = None
    threshold_exceeded: str | None = None
    stop_trigger_metadata: dict[str, Any] = Field(default_factory=dict)
