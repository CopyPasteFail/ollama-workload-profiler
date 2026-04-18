from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class VerdictLabel(StrEnum):
    GOOD_FIT = "good_fit"
    GOOD_FIT_WITH_CAUTION = "good_fit_with_caution"
    USE_WITH_CAUTION = "use_with_caution"
    USE_ONLY_FOR_SHORT_TASKS = "use_only_for_short_tasks"
    AVOID_ON_THIS_HARDWARE = "avoid_on_this_hardware"

    @property
    def display_label(self) -> str:
        return VERDICT_DISPLAY_LABELS[self]


VERDICT_DISPLAY_LABELS: dict[VerdictLabel, str] = {
    VerdictLabel.GOOD_FIT: "Good fit",
    VerdictLabel.GOOD_FIT_WITH_CAUTION: "Good fit with caution",
    VerdictLabel.USE_WITH_CAUTION: "Use with caution",
    VerdictLabel.USE_ONLY_FOR_SHORT_TASKS: "Use only for short tasks",
    VerdictLabel.AVOID_ON_THIS_HARDWARE: "Avoid on this hardware",
}


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: VerdictLabel
    rationale: str
    supporting_metrics: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


UseCaseSupportingTuple = tuple[
    int,
    int,
    int,
    int,
    int,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]

USE_CASE_TUPLE_PLANNED_RUNS = 0
USE_CASE_TUPLE_TOTAL_RUNS = 1
USE_CASE_TUPLE_COMPLETED_RUNS = 2
USE_CASE_TUPLE_FAILED_RUNS = 3
USE_CASE_TUPLE_STOPPED_RUNS = 4
USE_CASE_TUPLE_AVG_ELAPSED_MS = 5
USE_CASE_TUPLE_AVG_TOKENS_PER_SECOND = 6
USE_CASE_TUPLE_AVG_TTFT_MS = 7
USE_CASE_TUPLE_AVG_RAM_HEADROOM_GB = 8
USE_CASE_TUPLE_FAILURE_RATE = 9


class UseCaseMatrixCell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict_label: VerdictLabel | None = None
    supporting_tuple: UseCaseSupportingTuple = (0, 0, 0, 0, 0, None, None, None, None, None)
    missing_planned_context: bool = False
