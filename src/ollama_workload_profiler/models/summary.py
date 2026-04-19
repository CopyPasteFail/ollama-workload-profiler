from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .failures import FailureInfo
from .verdicts import UseCaseMatrixCell


SummaryMetricValue = str | int | float | bool | None


class ModelSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = ""
    metrics: dict[str, SummaryMetricValue] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ReportSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark_methodology_version: str | None = None
    executive_summary: str = ""
    session_metrics: dict[str, SummaryMetricValue] = Field(default_factory=dict)
    model_summaries: dict[str, ModelSummary] = Field(default_factory=dict)
    use_case_matrix: list[dict[str, str | UseCaseMatrixCell]] = Field(default_factory=list)
    benchmark_summaries: list[dict[str, SummaryMetricValue]] = Field(default_factory=list)
    phase_peak_summaries: list[dict[str, SummaryMetricValue]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    failures: list[FailureInfo] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
