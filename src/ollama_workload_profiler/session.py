from __future__ import annotations

import hashlib
import platform
import shutil
import socket
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from itertools import cycle
from pathlib import Path
from time import perf_counter
from typing import Any

import psutil

from .benchmarks import build_scenarios_for_benchmark
from .benchmarks.base import BenchmarkRunner, ExecutionMode, ExecutionRequest, ExecutionResult
from .env_check import detect_ollama_binary, detect_python_environment
from .metrics.process import find_ollama_processes
from .metrics.sampler import PollingProcessSampler
from .models.plan import BenchmarkSessionPlan, BenchmarkType, PlannedRun
from .models.results import RunResult, RunState
from .models.summary import ReportSummary
from .ollama_client import OllamaClient
from .prompts.scenarios import MultiTurnChatPromptPayload, ScenarioDefinition
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
            self._echo(self._finalize_live_line(_format_run_result_row(run_result, total_runs=total_runs)))
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
    BenchmarkType.STRESS,
)


def expand_session_plan(plan: BenchmarkSessionPlan) -> list[PlannedRun]:
    ordered_benchmark_types = _ordered_benchmark_types(plan.benchmark_types)
    runs: list[PlannedRun] = []
    run_index = 1

    for context_index, context_size in enumerate(plan.contexts, start=1):
        for benchmark_type_index, benchmark_type in enumerate(ordered_benchmark_types, start=1):
            scenarios = _build_scenarios_for_benchmark(benchmark_type, context_size)
            for scenario_index, scenario in enumerate(scenarios, start=1):
                for repetition_index in range(1, plan.repetitions + 1):
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


class _OllamaDispatcher:
    def __init__(self, client: OllamaClient) -> None:
        self._client = client

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

        if isinstance(prompt_payload, MultiTurnChatPromptPayload):
            response = self._client.chat(
                model=request.run.model_name,
                messages=[{"role": "user", "content": turn} for turn in prompt_payload.turns],
                options=self._build_options(request),
            )
        else:
            response = self._client.generate(
                model=request.run.model_name,
                prompt=prompt_payload.text,
                options=self._build_options(request),
            )

        elapsed_ms = _response_elapsed_ms(response, started_at)
        metrics = _response_metrics(response)
        return ExecutionResult(elapsed_ms=elapsed_ms, metrics=metrics)

    def _build_options(self, request: ExecutionRequest) -> dict[str, Any]:
        options: dict[str, Any] = {
            "num_ctx": request.run.context_size,
            "num_predict": request.scenario.target_output_tokens,
        }
        if request.execution_mode is ExecutionMode.TTFT:
            # Some Ollama models collapse `num_predict=1` into a terminal metadata
            # chunk with no observable streamed token, so TTFT uses a small floor.
            options["num_predict"] = max(8, request.scenario.target_output_tokens)
        return options

    def _execute_ttft_stream(
        self,
        *,
        request: ExecutionRequest,
        prompt_payload: Any,
        started_at: float,
    ) -> tuple[dict[str, Any], dict[str, Any], float]:
        if isinstance(prompt_payload, MultiTurnChatPromptPayload):
            stream_factory = lambda: self._client.stream_chat(
                model=request.run.model_name,
                messages=[{"role": "user", "content": turn} for turn in prompt_payload.turns],
                options=self._build_options(request),
            )
        else:
            stream_factory = lambda: self._client.stream_generate(
                model=request.run.model_name,
                prompt=prompt_payload.text,
                options=self._build_options(request),
            )

        first_token_at: float | None = None
        last_chunk: dict[str, Any] = {}
        stream_resource = stream_factory()
        stream_context = stream_resource if hasattr(stream_resource, "__enter__") else nullcontext(stream_resource)
        with stream_context as stream:
            for chunk in stream:
                last_chunk = dict(chunk)
                if first_token_at is None and _stream_chunk_has_text(chunk):
                    first_token_at = perf_counter()

        finished_at = perf_counter()
        metrics: dict[str, Any] = {
            "ttft_measurement_started_at": round(started_at, 6),
            "ttft_first_token_received": first_token_at is not None,
            "ttft_first_token_at": round(first_token_at, 6) if first_token_at is not None else None,
        }
        if first_token_at is not None:
            metrics["ttft_ms"] = round((first_token_at - started_at) * 1000, 3)

        return last_chunk, metrics, finished_at


def build_profile_session_plan(
    *,
    model_name: str,
    contexts: list[int],
    benchmark_types: list[BenchmarkType],
    repetitions: int = 1,
) -> BenchmarkSessionPlan:
    return BenchmarkSessionPlan(
        model_name=model_name,
        contexts=contexts,
        benchmark_types=benchmark_types,
        repetitions=repetitions,
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
        "repetitions": plan.repetitions,
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
    runner = BenchmarkRunner(
        dispatcher=_OllamaDispatcher(client),
        sampler_factory=lambda: PollingProcessSampler(process_finder=find_ollama_processes),
    )

    environment = _build_environment_snapshot(
        plan=plan,
        available_models=available_models if available_models is not None else client.list_models(),
        session_timestamp=timestamp,
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
    for planned_run in planned_runs:
        reporter.on_run_started(planned_run, total_runs=len(planned_runs))
        run_system_snapshot = _build_run_system_snapshot()
        run_result = runner.run(planned_run)
        run_result = run_result.model_copy(update={"system_snapshot": run_system_snapshot})
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
) -> dict[str, Any]:
    python_status = detect_python_environment()
    return {
        "session_started_at": session_timestamp.astimezone(timezone.utc).isoformat(),
        "python_version": python_status.python_version,
        "in_venv": python_status.in_venv,
        "executable": python_status.executable,
        "ollama_binary_found": detect_ollama_binary(),
        "host": _build_host_metadata(),
        "selected_model": plan.model_name,
        "available_models": available_models,
        "contexts": plan.contexts,
        "benchmark_types": [benchmark_type.value for benchmark_type in plan.benchmark_types],
        "repetitions": plan.repetitions,
    }


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
    for key in ("prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration"):
        value = response.get(key)
        if isinstance(value, (int, float)):
            metrics[key] = value

    eval_count = response.get("eval_count")
    eval_duration = response.get("eval_duration")
    if isinstance(eval_count, (int, float)) and isinstance(eval_duration, (int, float)) and eval_duration:
        metrics["tokens_per_second"] = round(float(eval_count) / (float(eval_duration) / 1_000_000_000), 3)

    return metrics


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
    return {
        "cpu_percent": _safe_cpu_percent(),
        "memory_available_mb": _bytes_to_mb(getattr(virtual_memory, "available", None)),
        "memory_used_percent": _safe_number(getattr(virtual_memory, "percent", None)),
        "ollama_process_count": len(find_ollama_processes()),
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
    return f"{int(round(float(value)))}s"


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


def _bytes_to_mb(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return round(float(value) / (1024 * 1024), 3)


def _safe_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return round(float(value), 3)


def _safe_string(factory: Any) -> str | None:
    try:
        value = factory()
    except Exception:
        return None
    if not isinstance(value, str):
        return None
    return value or None
