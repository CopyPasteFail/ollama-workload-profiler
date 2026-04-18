from __future__ import annotations

import hashlib
from contextlib import nullcontext
from dataclasses import dataclass
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from .benchmarks import build_scenarios_for_benchmark
from .benchmarks.base import BenchmarkRunner, ExecutionMode, ExecutionRequest, ExecutionResult
from .env_check import detect_ollama_binary, detect_python_environment
from .metrics.process import find_ollama_processes
from .metrics.sampler import PollingProcessSampler
from .models.plan import BenchmarkSessionPlan, BenchmarkType, PlannedRun
from .models.results import RunResult
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
    for planned_run in planned_runs:
        run_result = runner.run(planned_run)
        runs.append(run_result)
        append_run_artifact(session_dir, run=run_result)

    summary = build_report_summary(plan=plan, environment=environment, runs=runs)
    report_markdown = render_markdown_report(summary)
    finalize_session_artifacts(
        session_dir,
        runs=runs,
        summary=summary,
        report_markdown=report_markdown,
    )

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
