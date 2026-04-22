from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from ..metrics.phases import compute_gpu_telemetry_summary, compute_phase_peaks, compute_process_telemetry_summary
from ..metrics.sampler import SamplePoint
from ..models.failures import FailureInfo
from ..models.plan import BenchmarkType, PlannedRun
from ..models.results import RunResult, RunState
from ..execution_settings import normalize_execution_settings
from ..prompts.scenarios import ScenarioDefinition


class ExecutionMode(StrEnum):
    GENERATE = "generate"
    TTFT = "ttft"
    CONCURRENCY = "concurrency"


@dataclass(frozen=True, slots=True)
class BenchmarkFamily:
    benchmark_type: BenchmarkType
    execution_mode: ExecutionMode
    scenarios: Callable[[int], list[ScenarioDefinition]]

    def resolve_scenarios(self, context_size: int) -> list[ScenarioDefinition]:
        return self.scenarios(context_size)


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    run: PlannedRun
    scenario: ScenarioDefinition
    execution_mode: ExecutionMode
    execution_settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_settings",
            normalize_execution_settings(self.execution_settings),
        )


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    elapsed_ms: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class BenchmarkExecutionStopped(RuntimeError):
    """Raised when benchmark execution is interrupted intentionally."""


class BenchmarkDispatcher(Protocol):
    def execute(self, request: ExecutionRequest) -> ExecutionResult: ...


class BenchmarkSampler(Protocol):
    def start(self, run: PlannedRun) -> None: ...

    def stop(self) -> Sequence[SamplePoint | Mapping[str, Any]]: ...


class BenchmarkRunner:
    def __init__(
        self,
        *,
        dispatcher: BenchmarkDispatcher,
        sampler_factory: Callable[[], BenchmarkSampler],
        family_resolver: Callable[[BenchmarkType], BenchmarkFamily] | None = None,
        on_state_change: Callable[[PlannedRun, RunState], None] | None = None,
        execution_settings: Mapping[str, Any] | None = None,
    ) -> None:
        if family_resolver is None:
            from . import resolve_benchmark_family

            family_resolver = resolve_benchmark_family
        self._dispatcher = dispatcher
        self._sampler_factory = sampler_factory
        self._family_resolver = family_resolver
        self._on_state_change = on_state_change
        self._execution_settings = normalize_execution_settings(execution_settings)

    def run(self, planned_run: PlannedRun) -> RunResult:
        family = self._family_resolver(planned_run.benchmark_type)
        scenario = self._resolve_scenario(family, planned_run)
        sampler = self._sampler_factory()
        self._emit_state(planned_run, RunState.STARTING)

        sampler_started = False

        try:
            sampler.start(planned_run)
            sampler_started = True
            outcome = self._dispatcher.execute(
                ExecutionRequest(
                    run=planned_run,
                    scenario=scenario,
                    execution_mode=family.execution_mode,
                    execution_settings=dict(self._execution_settings),
                )
            )
        except BenchmarkExecutionStopped as exc:
            result = self._finalize_terminal_result(
                planned_run=planned_run,
                scenario=scenario,
                execution_state=RunState.STOPPED,
                elapsed_ms=None,
                metrics={"stop_reason": str(exc)},
                sampler=sampler,
                sampler_started=sampler_started,
            )
            self._emit_state(planned_run, result.state)
            return result
        except Exception as exc:
            result = self._finalize_terminal_result(
                planned_run=planned_run,
                scenario=scenario,
                execution_state=RunState.FAILED,
                elapsed_ms=None,
                metrics={
                    "error": str(exc),
                    "exception_class": exc.__class__.__name__,
                },
                sampler=sampler,
                sampler_started=sampler_started,
            )
            self._emit_state(planned_run, result.state)
            return result

        result = self._finalize_terminal_result(
            planned_run=planned_run,
            scenario=scenario,
            execution_state=RunState.COMPLETED,
            elapsed_ms=outcome.elapsed_ms,
            metrics=outcome.metrics,
            sampler=sampler,
            sampler_started=sampler_started,
        )
        self._emit_state(planned_run, result.state)
        return result

    def _resolve_scenario(
        self, family: BenchmarkFamily, planned_run: PlannedRun
    ) -> ScenarioDefinition:
        for scenario in family.resolve_scenarios(planned_run.context_size):
            if scenario.scenario_id == planned_run.scenario_id:
                return scenario
        raise ValueError(
            f"Unknown scenario {planned_run.scenario_id!r} for benchmark "
            f"{planned_run.benchmark_type.value}"
        )

    def _finalize(
        self,
        *,
        planned_run: PlannedRun,
        scenario: ScenarioDefinition,
        state: RunState,
        elapsed_ms: float | None,
        metrics: Mapping[str, Any],
        samples: Sequence[SamplePoint | Mapping[str, Any]],
    ) -> RunResult:
        finalized_metrics = dict(metrics)
        finalized_metrics.update(_scenario_metadata(planned_run.benchmark_type, scenario))
        if samples:
            finalized_metrics["phase_peaks"] = {
                phase: {
                    "phase": peak.phase,
                    "rss_mb": peak.rss_mb,
                    "cpu_percent": peak.cpu_percent,
                }
                for phase, peak in compute_phase_peaks(samples).items()
            }
            finalized_metrics.update(compute_process_telemetry_summary(samples))
            finalized_metrics.update(compute_gpu_telemetry_summary(samples))

        failure: FailureInfo | None = None
        partial_result = False
        if state is RunState.FAILED:
            failure_message = (
                finalized_metrics.get("finalization_error")
                or finalized_metrics.get("error")
                or "Run did not complete successfully."
            )
            failure = FailureInfo(
                kind=state.value,
                phase="execution",
                message=str(failure_message),
                exception_class=_optional_exception_class(finalized_metrics),
            )
            partial_result = elapsed_ms is not None or bool(samples)
        elif state is RunState.STOPPED:
            failure_message = finalized_metrics.get("stop_reason") or "Run stopped before completion."
            failure = FailureInfo(
                kind=state.value,
                phase="execution",
                message=str(failure_message),
            )

        return RunResult(
            run_id=planned_run.run_id,
            run_index=planned_run.run_index,
            model_name=planned_run.model_name,
            context_size=planned_run.context_size,
            context_index=planned_run.context_index,
            benchmark_type=planned_run.benchmark_type,
            benchmark_type_index=planned_run.benchmark_type_index,
            scenario_id=planned_run.scenario_id,
            scenario_index=planned_run.scenario_index,
            scenario_version=planned_run.scenario_version or scenario.version,
            scenario_name=planned_run.scenario_name or scenario.name,
            repetition_index=planned_run.repetition_index,
            state=state,
            elapsed_ms=elapsed_ms,
            partial_result=partial_result,
            failure=failure,
            metrics=finalized_metrics,
        )

    def _finalize_terminal_result(
        self,
        *,
        planned_run: PlannedRun,
        scenario: ScenarioDefinition,
        execution_state: RunState,
        elapsed_ms: float | None,
        metrics: Mapping[str, Any],
        sampler: BenchmarkSampler,
        sampler_started: bool,
    ) -> RunResult:
        finalized_metrics = dict(metrics)
        finalized_metrics["execution_outcome_state"] = execution_state.value

        try:
            samples = self._stop_sampler(sampler, sampler_started)
            return self._finalize(
                planned_run=planned_run,
                scenario=scenario,
                state=execution_state,
                elapsed_ms=elapsed_ms,
                metrics=finalized_metrics,
                samples=samples,
            )
        except Exception as exc:
            fallback_metrics = dict(finalized_metrics)
            fallback_metrics.update(
                _scenario_metadata(planned_run.benchmark_type, scenario)
            )
            return RunResult(
                run_id=planned_run.run_id,
                run_index=planned_run.run_index,
                model_name=planned_run.model_name,
                context_size=planned_run.context_size,
                context_index=planned_run.context_index,
                benchmark_type=planned_run.benchmark_type,
                benchmark_type_index=planned_run.benchmark_type_index,
                scenario_id=planned_run.scenario_id,
                scenario_index=planned_run.scenario_index,
                scenario_version=planned_run.scenario_version or scenario.version,
                scenario_name=planned_run.scenario_name or scenario.name,
                repetition_index=planned_run.repetition_index,
                state=RunState.FAILED,
                elapsed_ms=elapsed_ms,
                partial_result=elapsed_ms is not None or bool(finalized_metrics),
                failure=FailureInfo(
                    kind=RunState.FAILED.value,
                    phase="finalization",
                    message=str(exc),
                    exception_class=exc.__class__.__name__,
                ),
                metrics={
                    **fallback_metrics,
                    "finalization_error": str(exc),
                },
            )

    def _emit_state(self, planned_run: PlannedRun, state: RunState) -> None:
        if self._on_state_change is not None:
            self._on_state_change(planned_run, state)

    def _stop_sampler(
        self, sampler: BenchmarkSampler, sampler_started: bool
    ) -> Sequence[SamplePoint | Mapping[str, Any]]:
        if not sampler_started:
            return []
        return sampler.stop()


def _optional_exception_class(metrics: Mapping[str, Any]) -> str | None:
    exception_class = metrics.get("exception_class")
    if exception_class is None:
        return None
    return str(exception_class)


def _scenario_metadata(
    benchmark_type: BenchmarkType,
    scenario: ScenarioDefinition,
) -> dict[str, Any]:
    if benchmark_type is not BenchmarkType.COLD_WARM:
        return {}

    requested_behavior = scenario.profile_tag or "unspecified"
    return {
        "requested_prep_behavior": requested_behavior,
        "actual_prep_method": "scenario_declared",
    }
