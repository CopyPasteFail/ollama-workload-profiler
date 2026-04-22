from __future__ import annotations

import hashlib
import platform
import shutil
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from itertools import cycle
from math import ceil
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import psutil

from .benchmarks import build_scenarios_for_benchmark, resolve_benchmark_family
from .benchmarks.base import BenchmarkRunner, ExecutionMode, ExecutionRequest, ExecutionResult
from .env_check import (
    detect_accelerator_metadata,
    detect_ollama_binary,
    detect_python_environment,
)
from .metrics.process import find_ollama_processes
from .metrics.sampler import PollingProcessSampler
from .models.plan import BenchmarkSessionPlan, BenchmarkType, PlannedRun
from .models.results import RunResult, RunState
from .models.summary import ReportSummary
from .ollama_client import OllamaClient
from .prompts.scenarios import MultiTurnChatPromptPayload, ScenarioDefinition, TextPromptPayload, _repeat_to_length
from .reporting import (
    append_run_artifact,
    build_report_summary,
    finalize_session_artifacts,
    initialize_session_artifacts,
    render_markdown_report,
)


class ProgressReporter:
    def on_run_started(self, planned_run: PlannedRun, *, total_runs: int) -> None:
        return None

    def on_run_finished(self, run_result: RunResult, *, total_runs: int) -> None:
        return None

    def on_session_finished(self, summary: ReportSummary, *, total_runs: int) -> None:
        return None


class TerminalProgressReporter(ProgressReporter):
    def __init__(
        self,
        *,
        echo: Any,
        live: bool = False,
        poll_interval_seconds: float = 1.0,
        telemetry_provider: Any | None = None,
        clock: Any = perf_counter,
        terminal_width_provider: Any | None = None,
    ) -> None:
        self._echo = echo
        self._live = live
        self._poll_interval_seconds = poll_interval_seconds
        self._telemetry_provider = telemetry_provider or _build_live_telemetry_snapshot
        self._clock = clock
        self._terminal_width_provider = terminal_width_provider or _default_terminal_width
        self._completed_runs = 0
        self._failed_runs = 0
        self._stopped_runs = 0
        self._current_run: PlannedRun | None = None
        self._current_total_runs = 0
        self._current_run_started_at = 0.0
        self._live_stop_event = threading.Event()
        self._live_thread: threading.Thread | None = None
        self._live_spinner = cycle("|/-\\")
        self._lock = threading.Lock()
        self._last_live_width = 0

    def on_run_started(self, planned_run: PlannedRun, *, total_runs: int) -> None:
        with self._lock:
            self._current_run = planned_run
            self._current_total_runs = total_runs
            self._current_run_started_at = float(self._clock())

        self._echo(_format_run_header(planned_run, total_runs=total_runs))
        if not self._live:
            return None

        self._emit_live_status()
        self._start_live_thread()
        return None

    def on_run_finished(self, run_result: RunResult, *, total_runs: int) -> None:
        self._stop_live_thread()
        with self._lock:
            if run_result.state is RunState.COMPLETED:
                self._completed_runs += 1
            elif run_result.state is RunState.FAILED:
                self._failed_runs += 1
            elif run_result.state is RunState.STOPPED:
                self._stopped_runs += 1
            self._current_run = None
            self._current_total_runs = total_runs

        if self._live:
            self._echo(self._finalize_live_line(_format_run_result_row_live(run_result, total_runs=total_runs)))
        return None

    def on_session_finished(self, summary: ReportSummary, *, total_runs: int) -> None:
        if not self._live:
            return None

        self._echo(
            self._finalize_live_line(
                "Session summary"
                f" | completed {summary.session_metrics['completed_runs']}"
                f" | failed {summary.session_metrics['failed_runs']}"
                f" | total {total_runs}"
            )
        )
        return None

    def _start_live_thread(self) -> None:
        self._stop_live_thread()
        self._live_stop_event.clear()
        self._live_thread = threading.Thread(target=self._poll_live_status, daemon=True)
        self._live_thread.start()

    def _stop_live_thread(self) -> None:
        self._live_stop_event.set()
        if self._live_thread is not None:
            self._live_thread.join(timeout=max(self._poll_interval_seconds, 0.1))
            self._live_thread = None

    def _poll_live_status(self) -> None:
        while not self._live_stop_event.wait(self._poll_interval_seconds):
            self._emit_live_status()

    def _emit_live_status(self) -> None:
        with self._lock:
            current_run = self._current_run
            total_runs = self._current_total_runs
            started_at = self._current_run_started_at
            completed_runs = self._completed_runs
            failed_runs = self._failed_runs
            stopped_runs = self._stopped_runs

        if current_run is None or total_runs <= 0:
            return None

        telemetry = dict(self._telemetry_provider())
        elapsed_seconds = max(0.0, float(self._clock()) - started_at)
        self._echo(
            self._render_live_line(
                _format_live_status_line(
                    current_run,
                    total_runs=total_runs,
                    elapsed_seconds=elapsed_seconds,
                    telemetry=telemetry,
                    spinner=next(self._live_spinner),
                    completed_runs=completed_runs,
                    failed_runs=failed_runs + stopped_runs,
                )
            ),
            nl=False,
        )
        return None

    def _render_live_line(self, message: str) -> str:
        terminal_width = max(1, int(self._terminal_width_provider()))
        visible_width = max(1, terminal_width - 1)
        clipped_message = _truncate_to_width(message, visible_width)
        padded_message = clipped_message
        if len(clipped_message) < self._last_live_width:
            padded_message = clipped_message + (" " * (self._last_live_width - len(clipped_message)))
        self._last_live_width = max(self._last_live_width, len(clipped_message))
        return "\r" + padded_message

    def _finalize_live_line(self, message: str) -> str:
        finalized_message = self._render_live_line(message)
        self._last_live_width = 0
        return finalized_message
BENCHMARK_EXECUTION_ORDER: tuple[BenchmarkType, ...] = (
    BenchmarkType.SMOKE,
    BenchmarkType.COLD_WARM,
    BenchmarkType.PROMPT_SCALING,
    BenchmarkType.CONTEXT_SCALING,
    BenchmarkType.OUTPUT_SCALING,
    BenchmarkType.USE_CASE_PROFILES,
    BenchmarkType.TTFT,
    BenchmarkType.CONCURRENCY_SMOKE,
    BenchmarkType.STRESS,
)

CALIBRATION_MAX_ATTEMPTS = 4
CALIBRATION_TOLERANCE_RATIO = 0.05


def expand_session_plan(plan: BenchmarkSessionPlan) -> list[PlannedRun]:
    ordered_benchmark_types = _ordered_benchmark_types(plan.benchmark_types)
    runs: list[PlannedRun] = []
    run_index = 1
    repetitions = _requested_repetitions(plan)

    for context_index, context_size in enumerate(plan.contexts, start=1):
        for benchmark_type_index, benchmark_type in enumerate(ordered_benchmark_types, start=1):
            scenarios = _build_scenarios_for_benchmark(benchmark_type, context_size)
            for scenario_index, scenario in enumerate(scenarios, start=1):
                for repetition_index in range(1, repetitions + 1):
                    run_id = _build_run_id(
                        model_name=plan.model_name,
                        context_size=context_size,
                        context_index=context_index,
                        benchmark_type=benchmark_type,
                        benchmark_type_index=benchmark_type_index,
                        scenario_id=scenario.scenario_id,
                        scenario_index=scenario_index,
                        repetition_index=repetition_index,
                    )
                    runs.append(
                        PlannedRun(
                            run_id=run_id,
                            run_index=run_index,
                            model_name=plan.model_name,
                            context_size=context_size,
                            context_index=context_index,
                            benchmark_type=benchmark_type,
                            benchmark_type_index=benchmark_type_index,
                            scenario_id=scenario.scenario_id,
                            scenario_index=scenario_index,
                            repetition_index=repetition_index,
                            scenario_name=scenario.name,
                            scenario_version=scenario.version,
                        )
                    )
                    run_index += 1

    return runs


@dataclass(frozen=True, slots=True)
class ProfileSessionResult:
    session_dir: Path
    plan: BenchmarkSessionPlan
    expanded_plan: list[PlannedRun]
    runs: list[RunResult]
    summary: ReportSummary


def _build_execution_options(
    *,
    planned_run: PlannedRun,
    scenario: ScenarioDefinition,
    execution_settings: Mapping[str, Any],
    execution_mode: ExecutionMode,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "num_ctx": planned_run.context_size,
        "num_predict": scenario.target_output_tokens,
        "seed": execution_settings["seed"],
        "temperature": execution_settings["temperature"],
    }

    top_p = execution_settings.get("top_p")
    if top_p is not None:
        options["top_p"] = top_p

    if execution_mode is ExecutionMode.TTFT:
        # Some Ollama models collapse `num_predict=1` into a terminal metadata
        # chunk with no observable streamed token, so TTFT uses a small floor.
        options["num_predict"] = max(8, scenario.target_output_tokens)

    return options


def _calibration_cache_key(planned_run: PlannedRun, scenario: ScenarioDefinition) -> tuple[str, int, str, str]:
    return (
        planned_run.model_name,
        planned_run.context_size,
        scenario.scenario_id,
        scenario.prompt_template_version or scenario.version or "v1",
    )


def _within_calibration_tolerance(actual_prompt_tokens: int, target_prompt_tokens: int) -> bool:
    if target_prompt_tokens <= 0:
        return False
    tolerance = max(1, int(round(target_prompt_tokens * CALIBRATION_TOLERANCE_RATIO)))
    return abs(actual_prompt_tokens - target_prompt_tokens) <= tolerance


def _next_calibration_candidate_length(
    candidate_length: int,
    actual_prompt_tokens: int,
    target_prompt_tokens: int,
) -> int:
    if actual_prompt_tokens <= 0:
        return max(candidate_length + target_prompt_tokens, candidate_length * 2, 1)

    next_length = int(round(candidate_length * (target_prompt_tokens / actual_prompt_tokens)))
    return max(1, next_length)


def _calibrate_context_prompt(
    *,
    client: OllamaClient,
    planned_run: PlannedRun,
    scenario: ScenarioDefinition,
    execution_settings: Mapping[str, Any],
    options: Mapping[str, Any],
    calibration_cache: dict[tuple[str, int, str, str], dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(scenario.prompt_payload, TextPromptPayload):
        return {
            "prompt": scenario.prompt_payload,
            "requested_fill_ratio": scenario.fill_ratio,
            "target_prompt_tokens": scenario.target_prompt_tokens,
            "actual_prompt_tokens": None,
            "calibration_status": "failed",
            "calibration_attempts": 0,
            "calibration_cache_hit": False,
        }

    cache_key = _calibration_cache_key(planned_run, scenario)
    cached_entry = calibration_cache.get(cache_key)
    if cached_entry is not None:
        return {
            **cached_entry,
            "calibration_cache_hit": True,
        }

    target_prompt_tokens = scenario.target_prompt_tokens or planned_run.context_size
    base_text = scenario.prompt_payload.text
    candidate_length = max(1, target_prompt_tokens)
    probe_options = dict(options)
    probe_options["num_predict"] = 1

    calibration_status = "failed"
    actual_prompt_tokens: int | None = None
    prompt = _repeat_to_length(base_text, candidate_length)
    attempts = 0

    for attempt in range(1, CALIBRATION_MAX_ATTEMPTS + 1):
        attempts = attempt
        response = client.generate(
            model=planned_run.model_name,
            prompt=prompt,
            options=probe_options,
        )
        actual_prompt_value = response.get("prompt_eval_count")
        if not isinstance(actual_prompt_value, (int, float)) or isinstance(actual_prompt_value, bool):
            calibration_status = "failed"
            break

        actual_prompt_tokens = int(actual_prompt_value)
        if _within_calibration_tolerance(actual_prompt_tokens, target_prompt_tokens):
            calibration_status = "exact"
            break

        calibration_status = "approximate"
        candidate_length = _next_calibration_candidate_length(
            candidate_length,
            actual_prompt_tokens,
            target_prompt_tokens,
        )
        prompt = _repeat_to_length(base_text, candidate_length)

    cached_entry = {
        "prompt": prompt,
        "requested_fill_ratio": scenario.fill_ratio,
        "target_prompt_tokens": target_prompt_tokens,
        "actual_prompt_tokens": actual_prompt_tokens,
        "calibration_status": calibration_status,
        "calibration_attempts": attempts,
        "calibration_cache_hit": False,
    }
    calibration_cache[cache_key] = dict(cached_entry)
    return cached_entry


def _resolve_planned_scenario(planned_run: PlannedRun) -> ScenarioDefinition:
    for scenario in _build_scenarios_for_benchmark(planned_run.benchmark_type, planned_run.context_size):
        if scenario.scenario_id == planned_run.scenario_id:
            return scenario
    raise ValueError(
        f"Unknown scenario {planned_run.scenario_id!r} for benchmark "
        f"{planned_run.benchmark_type.value}"
    )


def _warmup_runs_requested(execution_settings: Mapping[str, Any]) -> int:
    warmup_runs = execution_settings.get("warmup_runs", 1)
    if isinstance(warmup_runs, bool) or not isinstance(warmup_runs, int) or warmup_runs < 1:
        raise ValueError("execution_settings.warmup_runs must be a positive integer")
    return warmup_runs


def _warmup_context_boundary(
    *,
    client: OllamaClient,
    planned_run: PlannedRun,
    scenario: ScenarioDefinition,
    execution_settings: Mapping[str, Any],
    execution_mode: ExecutionMode,
) -> bool:
    options = _build_execution_options(
        planned_run=planned_run,
        scenario=scenario,
        execution_settings=execution_settings,
        execution_mode=execution_mode,
    )

    prompt_payload = scenario.prompt_payload
    if isinstance(prompt_payload, MultiTurnChatPromptPayload):
        client.chat(
            model=planned_run.model_name,
            messages=[{"role": "user", "content": turn} for turn in prompt_payload.turns],
            options=options,
        )
    else:
        client.generate(
            model=planned_run.model_name,
            prompt=prompt_payload.text,
            options=options,
        )
    return True


def _prepare_run_preference(
    *,
    client: OllamaClient,
    planned_run: PlannedRun,
    scenario: ScenarioDefinition,
    execution_settings: Mapping[str, Any],
    warmed_contexts: set[tuple[str, int]],
) -> dict[str, Any]:
    boundary_key = (planned_run.model_name, planned_run.context_size)
    execution_mode = resolve_benchmark_family(planned_run.benchmark_type).execution_mode
    warmup_enabled = execution_settings.get("warmup_enabled", True)

    if planned_run.benchmark_type is BenchmarkType.COLD_WARM and scenario.profile_tag == "cold_start":
        try:
            client.unload_model(model=planned_run.model_name)
            return {
                "requested_prep_behavior": "cold_start",
                "actual_prep_method": "explicit_unload",
                "prep_enforcement_succeeded": True,
            }
        except Exception:
            return {
                "requested_prep_behavior": "cold_start",
                "actual_prep_method": "explicit_unload_failed",
                "prep_enforcement_succeeded": False,
            }

    if planned_run.benchmark_type is BenchmarkType.COLD_WARM and scenario.profile_tag == "warm_start":
        try:
            client.preload_model(
                model=planned_run.model_name,
                options=_build_execution_options(
                    planned_run=planned_run,
                    scenario=scenario,
                    execution_settings=execution_settings,
                    execution_mode=execution_mode,
                ),
            )
            return {
                "requested_prep_behavior": "warm_start",
                "actual_prep_method": "explicit_preload",
                "prep_enforcement_succeeded": True,
            }
        except Exception:
            return {
                "requested_prep_behavior": "warm_start",
                "actual_prep_method": "explicit_preload_failed",
                "prep_enforcement_succeeded": False,
            }

    if planned_run.benchmark_type is BenchmarkType.CONTEXT_SCALING:
        return {
            "requested_prep_behavior": "calibrated_context_fill",
            "actual_prep_method": "calibration",
            "prep_enforcement_succeeded": True,
        }

    if boundary_key in warmed_contexts:
        return {
            "requested_prep_behavior": "session_warmup",
            "actual_prep_method": "already_warm",
            "prep_enforcement_succeeded": True,
        }

    if not warmup_enabled:
        return {
            "requested_prep_behavior": "session_warmup",
            "actual_prep_method": "warmup_disabled",
            "prep_enforcement_succeeded": False,
        }

    warmup_method = "chat" if isinstance(scenario.prompt_payload, MultiTurnChatPromptPayload) else "generate"
    if not hasattr(client, warmup_method):
        return {
            "requested_prep_behavior": "session_warmup",
            "actual_prep_method": "warmup_unavailable",
            "prep_enforcement_succeeded": False,
        }

    warmup_runs = _warmup_runs_requested(execution_settings)
    warmup_success = True
    for _ in range(warmup_runs):
        try:
            _warmup_context_boundary(
                client=client,
                planned_run=planned_run,
                scenario=scenario,
                execution_settings=execution_settings,
                execution_mode=execution_mode,
            )
        except Exception:
            warmup_success = False
            break

    if warmup_success:
        warmed_contexts.add(boundary_key)
    return {
        "requested_prep_behavior": "session_warmup",
        "actual_prep_method": "session_warmup" if warmup_success else "session_warmup_failed",
        "prep_enforcement_succeeded": warmup_success,
        }


def _build_prep_eligibility_metadata(
    *,
    run_result: RunResult,
    prep_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    prep_succeeded = bool(prep_metadata.get("prep_enforcement_succeeded"))
    requested_behavior = prep_metadata.get("requested_prep_behavior")
    completed = run_result.state is RunState.COMPLETED
    metrics = run_result.metrics
    calibration_status = metrics.get("calibration_status")
    ttft_first_emission_ms = metrics.get("ttft_first_emission_ms", metrics.get("ttft_ms"))

    calibrated_context_eligible = bool(
        completed
        and calibration_status in {"exact", "approximate"}
    )
    strict_aggregate_eligible = bool(
        completed
        and (
            prep_succeeded
            or calibration_status == "approximate"
        )
        and calibration_status != "failed"
    )

    metadata: dict[str, Any] = {
        "eligible_for_strict_aggregate": strict_aggregate_eligible,
        "eligible_for_cold_start_aggregate": bool(
            completed and prep_succeeded and requested_behavior == "cold_start"
        ),
        "eligible_for_ttft_aggregate": bool(
            completed and isinstance(ttft_first_emission_ms, int | float)
        ),
        "eligible_for_calibrated_context_aggregate": calibrated_context_eligible,
    }
    return metadata


class _OllamaDispatcher:
    def __init__(
        self,
        client: OllamaClient,
        *,
        calibration_cache: dict[tuple[str, int, str, str], dict[str, Any]] | None = None,
        concurrency_client_factory: Any | None = None,
    ) -> None:
        self._client = client
        self._calibration_cache = calibration_cache if calibration_cache is not None else {}
        self._concurrency_client_factory = concurrency_client_factory if concurrency_client_factory is not None else OllamaClient

    def _build_options(self, request: ExecutionRequest) -> dict[str, Any]:
        return _build_execution_options(
            planned_run=request.run,
            scenario=request.scenario,
            execution_settings=request.execution_settings,
            execution_mode=request.execution_mode,
        )

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        prompt_payload = request.scenario.prompt_payload
        started_at = perf_counter()

        if request.execution_mode is ExecutionMode.TTFT:
            response, stream_metrics, finished_at = self._execute_ttft_stream(
                request=request,
                prompt_payload=prompt_payload,
                started_at=started_at,
            )
            elapsed_ms = _response_elapsed_ms(response, started_at, finished_at=finished_at)
            metrics = _response_metrics(response)
            metrics.update(stream_metrics)
            return ExecutionResult(elapsed_ms=elapsed_ms, metrics=metrics)

        if request.execution_mode is ExecutionMode.CONCURRENCY:
            metrics, finished_at = self._execute_concurrency_smoke(
                request=request,
                prompt_payload=prompt_payload,
            )
            elapsed_ms = round((finished_at - started_at) * 1000, 3)
            return ExecutionResult(elapsed_ms=elapsed_ms, metrics=metrics)

        options = _build_execution_options(
            planned_run=request.run,
            scenario=request.scenario,
            execution_settings=request.execution_settings,
            execution_mode=request.execution_mode,
        )
        calibration_metrics: dict[str, Any] = {}
        if (
            request.run.benchmark_type is BenchmarkType.CONTEXT_SCALING
            and isinstance(prompt_payload, TextPromptPayload)
        ):
            calibration = _calibrate_context_prompt(
                client=self._client,
                planned_run=request.run,
                scenario=request.scenario,
                execution_settings=request.execution_settings,
                options=options,
                calibration_cache=self._calibration_cache,
            )
            prompt = calibration.pop("prompt")
            calibration_metrics.update(calibration)
        else:
            prompt = prompt_payload.text if isinstance(prompt_payload, TextPromptPayload) else None

        if isinstance(prompt_payload, MultiTurnChatPromptPayload):
            response = self._client.chat(
                model=request.run.model_name,
                messages=[{"role": "user", "content": turn} for turn in prompt_payload.turns],
                options=options,
            )
        elif prompt is not None:
            response = self._client.generate(
                model=request.run.model_name,
                prompt=prompt,
                options=options,
            )
        else:
            raise TypeError(f"Unsupported prompt payload: {type(prompt_payload)!r}")

        elapsed_ms = _response_elapsed_ms(response, started_at)
        metrics = _response_metrics(response)
        if calibration_metrics:
            metrics.update(calibration_metrics)
            prompt_eval_count = metrics.get("prompt_eval_count")
            if isinstance(prompt_eval_count, (int, float)) and not isinstance(prompt_eval_count, bool):
                metrics["actual_prompt_tokens"] = int(prompt_eval_count)
        return ExecutionResult(elapsed_ms=elapsed_ms, metrics=metrics)

    def _execute_concurrency_smoke(
        self,
        *,
        request: ExecutionRequest,
        prompt_payload: Any,
    ) -> tuple[dict[str, Any], float]:
        if not isinstance(prompt_payload, TextPromptPayload):
            raise TypeError(f"Unsupported concurrency prompt payload: {type(prompt_payload)!r}")

        parallelism = request.scenario.parallelism
        if parallelism not in {2, 4}:
            raise ValueError("concurrency smoke supports parallelism values 2 and 4")

        start_barrier = threading.Barrier(parallelism)

        def execute_one(request_index: int) -> dict[str, Any]:
            start_barrier.wait()
            return self._execute_concurrency_stream_request(
                request=request,
                prompt=prompt_payload.text,
                request_index=request_index,
            )

        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            request_results = list(executor.map(execute_one, range(1, parallelism + 1)))

        metrics: dict[str, Any] = {
            "concurrency_mode": "same_machine_threaded_streams",
            "concurrency_parallelism": parallelism,
            "concurrency_request_count": len(request_results),
            "concurrency_requests": sorted(request_results, key=lambda item: item["request_index"]),
        }
        metrics.update(_build_concurrency_aggregate_metrics(request_results))
        return metrics, perf_counter()

    def _execute_concurrency_stream_request(
        self,
        *,
        request: ExecutionRequest,
        prompt: str,
        request_index: int,
    ) -> dict[str, Any]:
        started_at = perf_counter()
        first_token_at: float | None = None
        emission_offsets_ms: list[float] = []
        last_chunk: dict[str, Any] = {}
        worker_client = self._concurrency_client_factory()
        try:
            stream_resource = worker_client.stream_generate(
                model=request.run.model_name,
                prompt=prompt,
                options=self._build_options(request),
            )
            stream_context = stream_resource if hasattr(stream_resource, "__enter__") else nullcontext(stream_resource)
            with stream_context as stream:
                for chunk in stream:
                    last_chunk = dict(chunk)
                    if _stream_chunk_has_text(chunk):
                        emitted_at = perf_counter()
                        emission_offsets_ms.append(round((emitted_at - started_at) * 1000, 3))
                        if first_token_at is None:
                            first_token_at = emitted_at
        finally:
            close = getattr(worker_client, "close", None)
            if callable(close):
                close()

        finished_at = perf_counter()
        elapsed_ms = _response_elapsed_ms(last_chunk, started_at, finished_at=finished_at)
        result: dict[str, Any] = {
            "request_index": request_index,
            "elapsed_ms": elapsed_ms,
            "ttft_ms": round((first_token_at - started_at) * 1000, 3) if first_token_at is not None else None,
            "ttft_first_token_received": first_token_at is not None,
            **_build_stream_shape_metrics(emission_offsets_ms),
        }
        response_metrics = _response_metrics(last_chunk)
        if response_metrics:
            result["response_metrics"] = response_metrics
        return result

    def _execute_ttft_stream(
        self,
        *,
        request: ExecutionRequest,
        prompt_payload: Any,
        started_at: float,
    ) -> tuple[dict[str, Any], dict[str, Any], float]:
        if isinstance(prompt_payload, MultiTurnChatPromptPayload):
            def stream_factory() -> Any:
                return self._client.stream_chat(
                    model=request.run.model_name,
                    messages=[{"role": "user", "content": turn} for turn in prompt_payload.turns],
                    options=self._build_options(request),
                )
        else:
            def stream_factory() -> Any:
                return self._client.stream_generate(
                    model=request.run.model_name,
                    prompt=prompt_payload.text,
                    options=self._build_options(request),
                )

        first_token_at: float | None = None
        emission_offsets_ms: list[float] = []
        last_chunk: dict[str, Any] = {}
        stream_resource = stream_factory()
        stream_context = stream_resource if hasattr(stream_resource, "__enter__") else nullcontext(stream_resource)
        with stream_context as stream:
            for chunk in stream:
                last_chunk = dict(chunk)
                if _stream_chunk_has_text(chunk):
                    emitted_at = perf_counter()
                    emission_offsets_ms.append(round((emitted_at - started_at) * 1000, 3))
                    if first_token_at is None:
                        first_token_at = emitted_at

        finished_at = perf_counter()
        metrics: dict[str, Any] = {
            "ttft_measurement_started_at": round(started_at, 6),
            "ttft_first_token_received": first_token_at is not None,
            "ttft_first_token_at": round(first_token_at, 6) if first_token_at is not None else None,
            **_build_stream_shape_metrics(emission_offsets_ms),
        }
        if first_token_at is not None:
            metrics["ttft_ms"] = round((first_token_at - started_at) * 1000, 3)

        return last_chunk, metrics, finished_at


def build_profile_session_plan(
    *,
    model_name: str,
    contexts: list[int],
    benchmark_types: list[BenchmarkType],
    seed: int = 42,
    temperature: float = 0.0,
    top_p: float | None = None,
    repetitions: int = 1,
    execution_settings: dict[str, Any] | None = None,
) -> BenchmarkSessionPlan:
    settings = dict(execution_settings) if execution_settings is not None else {}
    settings.setdefault("seed", seed)
    settings.setdefault("temperature", temperature)
    settings.setdefault("repetitions", repetitions)
    if top_p is not None:
        settings.setdefault("top_p", top_p)
    return BenchmarkSessionPlan(
        model_name=model_name,
        contexts=contexts,
        benchmark_types=benchmark_types,
        execution_settings=settings,
    )


def summarize_session_budget(
    plan: BenchmarkSessionPlan,
    *,
    expanded_plan: list[PlannedRun] | None = None,
) -> dict[str, int | str | None]:
    planned_runs = list(expanded_plan) if expanded_plan is not None else expand_session_plan(plan)
    scenario_keys = {
        (run.context_size, run.benchmark_type.value, run.scenario_id)
        for run in planned_runs
    }
    run_count = len(planned_runs)
    return {
        "context_count": len(plan.contexts),
        "benchmark_type_count": len(plan.benchmark_types),
        "scenario_count": len(scenario_keys),
        "run_count": run_count,
        "repetitions": _requested_repetitions(plan),
        "warning": _budget_warning(run_count),
    }


def run_profile_session(
    *,
    plan: BenchmarkSessionPlan,
    client: OllamaClient,
    output_dir: Path,
    available_models: list[str] | None = None,
    expanded_plan: list[PlannedRun] | None = None,
    session_timestamp: datetime | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> ProfileSessionResult:
    session_root = Path(output_dir)
    session_root.mkdir(parents=True, exist_ok=True)

    timestamp = session_timestamp or datetime.now(timezone.utc)
    planned_runs = list(expanded_plan) if expanded_plan is not None else expand_session_plan(plan)
    calibration_cache: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    runner = BenchmarkRunner(
        dispatcher=_OllamaDispatcher(client, calibration_cache=calibration_cache),
        sampler_factory=lambda: PollingProcessSampler(process_finder=find_ollama_processes),
        execution_settings=plan.execution_settings,
    )

    environment = _build_environment_snapshot(
        plan=plan,
        available_models=available_models if available_models is not None else client.list_models(),
        session_timestamp=timestamp,
        client=client,
    )
    session_dir = initialize_session_artifacts(
        session_root,
        session_timestamp=timestamp,
        plan=plan,
        expanded_plan=planned_runs,
        environment=environment,
    )

    runs: list[RunResult] = []
    reporter = progress_reporter or ProgressReporter()
    warmed_contexts: set[tuple[str, int]] = set()
    for planned_run in planned_runs:
        reporter.on_run_started(planned_run, total_runs=len(planned_runs))
        scenario = _resolve_planned_scenario(planned_run)
        prep_metadata = _prepare_run_preference(
            client=client,
            planned_run=planned_run,
            scenario=scenario,
            execution_settings=plan.execution_settings,
            warmed_contexts=warmed_contexts,
        )
        run_system_snapshot = _build_run_system_snapshot()
        run_result = runner.run(planned_run)
        run_result = run_result.model_copy(
            update={
                "system_snapshot": run_system_snapshot,
                "metrics": {
                    **run_result.metrics,
                    **prep_metadata,
                    **_build_prep_eligibility_metadata(
                        run_result=run_result,
                        prep_metadata=prep_metadata,
                    ),
                },
            }
        )
        runs.append(run_result)
        append_run_artifact(session_dir, run=run_result)
        reporter.on_run_finished(run_result, total_runs=len(planned_runs))

    summary = build_report_summary(plan=plan, environment=environment, runs=runs)
    report_markdown = render_markdown_report(summary)
    finalize_session_artifacts(
        session_dir,
        runs=runs,
        summary=summary,
        report_markdown=report_markdown,
    )
    reporter.on_session_finished(summary, total_runs=len(planned_runs))

    return ProfileSessionResult(
        session_dir=session_dir,
        plan=plan,
        expanded_plan=planned_runs,
        runs=runs,
        summary=summary,
    )


def _build_environment_snapshot(
    *,
    plan: BenchmarkSessionPlan,
    available_models: list[str],
    session_timestamp: datetime,
    client: OllamaClient,
) -> dict[str, Any]:
    python_status = detect_python_environment()
    return {
        "session_started_at": session_timestamp.astimezone(timezone.utc).isoformat(),
        "benchmark_methodology_version": plan.benchmark_methodology_version,
        "python_version": python_status.python_version,
        "in_venv": python_status.in_venv,
        "executable": python_status.executable,
        "ollama_binary_found": detect_ollama_binary(),
        "host": _build_host_metadata(),
        "accelerator": detect_accelerator_metadata(),
        "ollama": _build_ollama_metadata(
            client=client,
            model_name=plan.model_name,
            available_models=available_models,
        ),
        "available_models": available_models,
        "execution_settings": dict(plan.execution_settings),
    }


def _build_ollama_metadata(
    *,
    client: OllamaClient,
    model_name: str,
    available_models: list[str],
) -> dict[str, Any]:
    version_value: str | None = None
    version_error: str | None = None
    try:
        version_value = client.version()
    except Exception as exc:
        version_error = str(exc) or exc.__class__.__name__

    show_payload: dict[str, Any] | None = None
    show_error: str | None = None
    try:
        show_payload = client.show_model(model_name)
    except Exception as exc:
        show_error = str(exc) or exc.__class__.__name__

    return {
        "binary_found": detect_ollama_binary(),
        "version": {
            "available": version_error is None and version_value is not None,
            "value": version_value,
            "error": version_error,
        },
        "selected_model": {
            "name": model_name,
            "show_available": show_error is None and show_payload is not None,
            "details": show_payload.get("details") if isinstance(show_payload, dict) else None,
            "model_info": show_payload.get("model_info") if isinstance(show_payload, dict) else None,
            "error": show_error,
        },
        "available_models": list(available_models),
    }


def _requested_repetitions(plan: BenchmarkSessionPlan) -> int:
    repetitions = plan.execution_settings.get("repetitions", 1)
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("execution_settings.repetitions must be a positive integer")
    return repetitions


def _response_elapsed_ms(
    response: dict[str, Any],
    started_at: float,
    *,
    finished_at: float | None = None,
) -> float:
    total_duration = response.get("total_duration")
    if isinstance(total_duration, (int, float)):
        return round(float(total_duration) / 1_000_000, 3)
    completed_at = perf_counter() if finished_at is None else finished_at
    return round((completed_at - started_at) * 1000, 3)


def _response_metrics(response: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in ("load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration"):
        value = response.get(key)
        if isinstance(value, (int, float)):
            metrics[key] = value

    load_duration = response.get("load_duration")
    if isinstance(load_duration, (int, float)):
        metrics["load_duration_ms"] = round(float(load_duration) / 1_000_000, 3)

    prompt_eval_count = response.get("prompt_eval_count")
    prompt_eval_duration = response.get("prompt_eval_duration")
    if (
        isinstance(prompt_eval_count, (int, float))
        and isinstance(prompt_eval_duration, (int, float))
        and prompt_eval_duration
    ):
        metrics["prompt_tokens_per_second"] = round(
            float(prompt_eval_count) / (float(prompt_eval_duration) / 1_000_000_000),
            3,
        )

    eval_count = response.get("eval_count")
    eval_duration = response.get("eval_duration")
    if isinstance(eval_count, (int, float)) and isinstance(eval_duration, (int, float)) and eval_duration:
        generation_tokens_per_second = round(
            float(eval_count) / (float(eval_duration) / 1_000_000_000),
            3,
        )
        metrics["generation_tokens_per_second"] = generation_tokens_per_second
        metrics["tokens_per_second"] = generation_tokens_per_second

    return metrics


def _build_stream_shape_metrics(emission_offsets_ms: list[float]) -> dict[str, Any]:
    emission_count = len(emission_offsets_ms)
    stream_duration_ms = (
        round(emission_offsets_ms[-1] - emission_offsets_ms[0], 3)
        if emission_count > 0
        else None
    )
    intervals_ms = [
        round(emission_offsets_ms[index] - emission_offsets_ms[index - 1], 3)
        for index in range(1, emission_count)
    ]

    return {
        "stream_emission_count": emission_count,
        "stream_emission_offsets_ms": list(emission_offsets_ms),
        "stream_duration_ms": stream_duration_ms,
        "stream_emission_interval_ms_median": _median_or_none(intervals_ms),
        "stream_emission_interval_ms_p95": _p95_or_none(intervals_ms),
        "stream_output_units_per_second": _stream_output_units_per_second(
            emission_count=emission_count,
            stream_duration_ms=stream_duration_ms,
        ),
        "stream_output_unit": "emission",
    }


def _stream_output_units_per_second(
    *,
    emission_count: int,
    stream_duration_ms: float | None,
) -> float | None:
    if emission_count < 2 or not isinstance(stream_duration_ms, (int, float)) or stream_duration_ms <= 0:
        return None
    return round(float(emission_count) / (float(stream_duration_ms) / 1000.0), 3)


def _build_concurrency_aggregate_metrics(request_results: list[Mapping[str, Any]]) -> dict[str, Any]:
    elapsed_values = _numeric_values([result.get("elapsed_ms") for result in request_results])
    ttft_values = _numeric_values([result.get("ttft_ms") for result in request_results])
    return {
        "concurrency_request_elapsed_ms_p50": _median_or_none(elapsed_values),
        "concurrency_request_elapsed_ms_p95": _p95_or_none(elapsed_values),
        "concurrency_request_ttft_ms_p50": _median_or_none(ttft_values),
        "concurrency_request_ttft_ms_p95": _p95_or_none(ttft_values),
    }


def _numeric_values(values: Iterable[Any]) -> list[float]:
    return [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(median(values)), 3)


def _p95_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    rank = max(1, ceil(len(ordered) * 0.95))
    return round(ordered[rank - 1], 3)


def _stream_chunk_has_text(chunk: Mapping[str, Any]) -> bool:
    # TTFT counts the first streamed model output, including "thinking" chunks,
    # because they are still the earliest token-bearing emission from the model.
    response_text = chunk.get("response")
    if isinstance(response_text, str) and response_text:
        return True

    thinking_text = chunk.get("thinking")
    if isinstance(thinking_text, str) and thinking_text:
        return True

    message = chunk.get("message")
    if isinstance(message, Mapping):
        message_content = message.get("content")
        if isinstance(message_content, str) and message_content:
            return True
        message_thinking = message.get("thinking")
        if isinstance(message_thinking, str) and message_thinking:
            return True

    return False


def _build_run_id(
    *,
    model_name: str,
    context_size: int,
    context_index: int,
    benchmark_type: BenchmarkType,
    benchmark_type_index: int,
    scenario_id: str,
    scenario_index: int,
    repetition_index: int,
) -> str:
    identity_source = "|".join(
        (
            model_name,
            str(context_size),
            str(context_index),
            benchmark_type.value,
            str(benchmark_type_index),
            scenario_id,
            str(scenario_index),
            str(repetition_index),
        )
    )
    digest = hashlib.sha256(identity_source.encode("utf-8")).hexdigest()
    return f"run-{digest[:12]}"


def _ordered_benchmark_types(
    benchmark_types: Iterable[BenchmarkType],
) -> list[BenchmarkType]:
    selected_types = set(benchmark_types)
    return [
        benchmark_type
        for benchmark_type in BENCHMARK_EXECUTION_ORDER
        if benchmark_type in selected_types
    ]


def _budget_warning(run_count: int) -> str | None:
    if run_count >= 20:
        return "Large plan selected; slower hardware may take a while."
    if run_count >= 10:
        return "Moderate plan selected; expect noticeable runtime on slower hardware."
    return None


def _build_scenarios_for_benchmark(
    benchmark_type: BenchmarkType, context_size: int
) -> list[ScenarioDefinition]:
    return build_scenarios_for_benchmark(benchmark_type, context_size)


def _build_host_metadata() -> dict[str, Any]:
    virtual_memory = _safe_virtual_memory()
    return {
        "hostname": _safe_string(socket.gethostname),
        "os": {
            "platform": _safe_string(platform.system),
            "release": _safe_string(platform.release),
            "version": _safe_string(platform.version),
            "machine": _safe_string(platform.machine),
        },
        "cpu": {
            "processor": _safe_string(platform.processor),
            "logical_cores": _safe_cpu_count(logical=True),
            "physical_cores": _safe_cpu_count(logical=False),
        },
        "memory": {
            "total_mb": _bytes_to_mb(getattr(virtual_memory, "total", None)),
        },
    }


def _build_run_system_snapshot() -> dict[str, Any]:
    virtual_memory = _safe_virtual_memory()
    cpu_percent = _safe_cpu_percent()
    available_memory_mb = _bytes_to_mb(getattr(virtual_memory, "available", None))
    memory_used_percent = _safe_number(getattr(virtual_memory, "percent", None))
    warning_reasons = _host_pressure_warning_reasons(
        system_cpu_load_snapshot=cpu_percent,
        available_system_memory_mb=available_memory_mb,
        memory_used_percent=memory_used_percent,
    )
    return {
        "cpu_percent": cpu_percent,
        "system_cpu_load_snapshot": cpu_percent,
        "memory_available_mb": available_memory_mb,
        "available_system_memory_mb": available_memory_mb,
        "memory_used_percent": memory_used_percent,
        "host_pressure_warning": bool(warning_reasons),
        "host_pressure_warning_reasons": warning_reasons,
        "ollama_process_count": _safe_ollama_process_count(),
    }


def _build_live_telemetry_snapshot() -> dict[str, float | None]:
    rss_total_mb = 0.0
    cpu_total = 0.0
    observed_process = False
    for process in find_ollama_processes():
        try:
            memory_info = process.memory_info()
            rss_total_mb += float(memory_info.rss) / (1024 * 1024)
            cpu_total += float(process.cpu_percent(interval=None))
            observed_process = True
        except Exception:
            continue

    return {
        "cpu_percent": round(cpu_total, 3) if observed_process else None,
        "rss_mb": round(rss_total_mb, 3) if observed_process else None,
        "tokens_per_second": None,
    }


def _format_run_header(planned_run: PlannedRun, *, total_runs: int) -> str:
    return (
        f"Run {planned_run.run_index}/{total_runs}"
        f" | {planned_run.benchmark_type.value}"
        f" | {planned_run.scenario_name or planned_run.scenario_id}"
        f" | ctx {planned_run.context_size}"
        f" | rep {planned_run.repetition_index}"
    )


def _format_live_status_line(
    planned_run: PlannedRun,
    *,
    total_runs: int,
    elapsed_seconds: float,
    telemetry: Mapping[str, Any],
    spinner: str,
    completed_runs: int,
    failed_runs: int,
) -> str:
    pending_runs = max(total_runs - completed_runs - failed_runs, 0)
    return (
        f"Live {spinner} {_render_progress_bar(completed_runs, total_runs)}"
        f" | Run {planned_run.run_index}/{total_runs}"
        f" | {planned_run.benchmark_type.value}"
        f" | {planned_run.scenario_name or planned_run.scenario_id}"
        f" | ctx {planned_run.context_size}"
        f" | rep {planned_run.repetition_index}"
        f" | elapsed {_format_elapsed_seconds_live(elapsed_seconds)}"
        f" | CPU {_format_stat(telemetry.get('cpu_percent'), suffix='%')}"
        f" | RSS {_format_stat(telemetry.get('rss_mb'), suffix=' MB')}"
        f" | tok/s {_format_stat(telemetry.get('tokens_per_second'))}"
        f" | completed {completed_runs} | pending {pending_runs} | failed {failed_runs}"
    )


def _format_run_result_row(run_result: RunResult, *, total_runs: int) -> str:
    return (
        "Result"
        f" | run {run_result.run_index}/{total_runs}"
        f" | {run_result.state.value}"
        f" | {run_result.benchmark_type.value}"
        f" | {run_result.scenario_name or run_result.scenario_id}"
        f" | ctx {run_result.context_size}"
        f" | rep {run_result.repetition_index}"
        f" | elapsed {_format_elapsed_seconds_result(run_result.elapsed_ms)}"
        f" | tok/s {_format_stat(run_result.metrics.get('tokens_per_second'))}"
    )


def _format_run_result_row_live(run_result: RunResult, *, total_runs: int) -> str:
    return (
        "Result"
        f" | run {run_result.run_index}/{total_runs}"
        f" | {run_result.state.value}"
        f" | {run_result.benchmark_type.value}"
        f" | {run_result.scenario_name or run_result.scenario_id}"
        f" | ctx {run_result.context_size}"
        f" | rep {run_result.repetition_index}"
        f" | elapsed {_format_elapsed_seconds_live(run_result.elapsed_ms / 1000.0)}"
        f" | tok/s {_format_stat(run_result.metrics.get('tokens_per_second'))}"
    )


def _render_progress_bar(completed_runs: int, total_runs: int, *, width: int = 10) -> str:
    if total_runs <= 0:
        return "[----------]"

    filled_width = int(width * completed_runs / total_runs)
    filled_width = max(0, min(width, filled_width))
    return "[" + ("#" * filled_width) + ("-" * (width - filled_width)) + "]"


def _format_stat(value: Any, *, suffix: str = "") -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "--"
    return f"{float(value):.1f}{suffix}"


def _format_elapsed_seconds_live(value: float | None) -> str:
    if not isinstance(value, (int, float)):
        return "--"

    total_seconds = max(0, int(round(float(value))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _format_elapsed_seconds_result(value_ms: float | None) -> str:
    if not isinstance(value_ms, (int, float)):
        return "--"
    return f"{float(value_ms) / 1000.0:.1f}s"


def _default_terminal_width() -> int:
    try:
        return int(shutil.get_terminal_size(fallback=(120, 24)).columns)
    except Exception:
        return 120


def _truncate_to_width(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


def _safe_virtual_memory() -> Any | None:
    try:
        return psutil.virtual_memory()
    except Exception:
        return None


def _safe_cpu_count(*, logical: bool) -> int | None:
    try:
        return psutil.cpu_count(logical=logical)
    except Exception:
        return None


def _safe_cpu_percent() -> float | None:
    try:
        return round(float(psutil.cpu_percent(interval=None)), 3)
    except Exception:
        return None


def _safe_ollama_process_count() -> int | None:
    try:
        return len(find_ollama_processes())
    except Exception:
        return None


def _bytes_to_mb(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return round(float(value) / (1024 * 1024), 3)


def _safe_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return round(float(value), 3)


def _host_pressure_warning_reasons(
    *,
    system_cpu_load_snapshot: float | None,
    available_system_memory_mb: float | None,
    memory_used_percent: float | None,
) -> list[str]:
    reasons: list[str] = []
    if isinstance(system_cpu_load_snapshot, (int, float)) and system_cpu_load_snapshot >= 80:
        reasons.append("system_cpu_load_snapshot >= 80%")
    if isinstance(memory_used_percent, (int, float)) and memory_used_percent >= 90:
        reasons.append("memory_used_percent >= 90%")
    if isinstance(available_system_memory_mb, (int, float)) and available_system_memory_mb < 1024:
        reasons.append("available_system_memory_mb < 1024")
    return reasons


def _safe_string(factory: Any) -> str | None:
    try:
        value = factory()
    except Exception:
        return None
    if not isinstance(value, str):
        return None
    return value or None
