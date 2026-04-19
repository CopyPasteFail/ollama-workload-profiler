from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys
from typing import Annotated

import typer

from .env_check import (
    detect_dependency_status,
    detect_ollama_binary,
    detect_python_environment,
    probe_ollama_server,
    summarize_doctor_status,
)
from .methodology import BENCHMARK_METHODOLOGY_VERSION
from .models.plan import BenchmarkType
from .ollama_client import OllamaClient
from .reporting.compare import (
    CompareArtifactError,
    compare_sessions,
    render_compare_json,
    render_compare_text,
)
from .session import (
    build_profile_session_plan,
    expand_session_plan,
    run_profile_session,
    summarize_session_budget,
    TerminalProgressReporter,
)

app = typer.Typer(help="Profile local Ollama workloads.")
_SUPPORTED_BENCHMARK_TYPES: tuple[BenchmarkType, ...] = tuple(BenchmarkType)


class CompareOutputFormat(str):
    TEXT = "text"
    JSON = "json"


def _build_terminal_echo() -> Callable[[str, bool], None]:
    def emit(message: str, nl: bool = True) -> None:
        sys.stdout.write(message + ("\n" if nl else ""))
        sys.stdout.flush()

    return emit


def _build_execution_settings(
    *,
    seed: int,
    temperature: float,
    top_p: float | None,
    repetitions: int,
    warmup_runs: int,
    warmup_enabled: bool,
) -> dict[str, object]:
    return {
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
        "repetitions": repetitions,
        "warmup_runs": warmup_runs,
        "warmup_enabled": warmup_enabled,
    }


def _format_policy_value(value: object) -> str:
    if value is None:
        return "unset"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _format_benchmark_policy(execution_settings: dict[str, object]) -> str:
    return (
        "Benchmark policy: "
        f"seed={_format_policy_value(execution_settings.get('seed'))}, "
        f"temperature={_format_policy_value(execution_settings.get('temperature'))}, "
        f"top_p={_format_policy_value(execution_settings.get('top_p'))}, "
        f"repetitions={_format_policy_value(execution_settings.get('repetitions'))}, "
        f"warmup_runs={_format_policy_value(execution_settings.get('warmup_runs'))}, "
        f"warmup_enabled={_format_policy_value(execution_settings.get('warmup_enabled'))}"
    )


def _parse_multi_select(
    raw_value: str,
    *,
    item_parser: Callable[[str], object],
    empty_message: str,
    invalid_message: Callable[[str], str],
    empty_token_message: str | None = None,
) -> list[object]:
    items = [segment.strip() for segment in raw_value.split(",")]
    if not any(items):
        raise typer.BadParameter(empty_message)

    parsed_items: list[object] = []
    seen: set[object] = set()

    for item in items:
        if not item:
            if empty_token_message is not None:
                raise typer.BadParameter(empty_token_message)
            raise typer.BadParameter(invalid_message(""))

        parsed_item = item_parser(item)
        if parsed_item in seen:
            continue

        seen.add(parsed_item)
        parsed_items.append(parsed_item)

    if not parsed_items:
        raise typer.BadParameter(empty_message)

    return parsed_items


def parse_contexts(raw_value: str) -> list[int]:
    """Parse a comma-separated context selection."""

    def parse_context(item: str) -> int:
        try:
            value = int(item)
        except ValueError as exc:
            raise typer.BadParameter("Contexts must be comma-separated integers.") from exc
        if value <= 0:
            raise typer.BadParameter("Contexts must be positive integers.")
        return value

    return _parse_multi_select(
        raw_value,
        item_parser=parse_context,
        empty_message="Select at least one context.",
        invalid_message=lambda _item: "Contexts must be comma-separated integers.",
    )


def parse_benchmark_types(raw_value: str) -> list[BenchmarkType]:
    """Parse a comma-separated benchmark type selection."""

    all_types = {benchmark_type.value: benchmark_type for benchmark_type in BenchmarkType}
    supported_types = {benchmark_type.value: benchmark_type for benchmark_type in _SUPPORTED_BENCHMARK_TYPES}

    def parse_benchmark_type(item: str) -> BenchmarkType:
        try:
            benchmark_type = all_types[item]
        except KeyError as exc:
            raise typer.BadParameter(f"Invalid benchmark type: {item}.") from exc

        if item not in supported_types:
            raise typer.BadParameter(f"Unsupported benchmark type: {item}.")

        return benchmark_type

    return _parse_multi_select(
        raw_value,
        item_parser=parse_benchmark_type,
        empty_message="Select at least one benchmark type.",
        invalid_message=lambda item: f"Invalid benchmark type: {item}.",
        empty_token_message="Benchmark types must not contain empty values.",
    )


@app.command()
def doctor() -> None:
    """Check local environment readiness."""
    python_status = detect_python_environment()
    dependencies_ok, missing_dependencies = detect_dependency_status()
    binary_found = detect_ollama_binary()
    reachable, models = probe_ollama_server() if binary_found else (False, [])
    summary = summarize_doctor_status(
        binary_found=binary_found,
        reachable=reachable,
        models=models,
        dependencies_ok=dependencies_ok,
        missing_dependencies=missing_dependencies,
    )

    typer.echo(f"Platform: {summary.platform_name}")
    typer.echo(f"Python: {python_status.python_version}")
    typer.echo(f"Venv: {'yes' if python_status.in_venv else 'no'}")
    typer.echo(f"Executable: {python_status.executable}")

    for message in summary.messages:
        typer.echo(message)

    if summary.remediation_hints:
        typer.echo("Remediation hints:")
        for hint in summary.remediation_hints:
            typer.echo(f"- {hint}")

    raise typer.Exit(code=summary.exit_code)


@app.command()
def compare(
    baseline_session_dir: Annotated[
        Path,
        typer.Argument(help="Baseline completed benchmark session directory."),
    ],
    candidate_session_dir: Annotated[
        Path,
        typer.Argument(help="Candidate completed benchmark session directory."),
    ],
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: text or json."),
    ] = CompareOutputFormat.TEXT,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Write comparison output to this file."),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Return nonzero for strict-blocking comparability issues."),
    ] = False,
    all_metrics: Annotated[
        bool,
        typer.Option("--all-metrics", help="Show unchanged and unavailable metrics in text output."),
    ] = False,
) -> None:
    """Compare two completed benchmark sessions."""
    if output_format not in {CompareOutputFormat.TEXT, CompareOutputFormat.JSON}:
        typer.echo("Invalid --format. Use 'text' or 'json'.")
        raise typer.Exit(code=2)

    try:
        result = compare_sessions(
            baseline_dir=baseline_session_dir,
            candidate_dir=candidate_session_dir,
        )
    except CompareArtifactError as exc:
        typer.echo(f"Compare failed: {exc}")
        raise typer.Exit(code=1) from exc

    rendered = (
        render_compare_json(result)
        if output_format == CompareOutputFormat.JSON
        else render_compare_text(result, all_metrics=all_metrics)
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        typer.echo(f"Comparison written to: {output}")
    else:
        typer.echo(rendered, nl=False)

    if strict and result.strict_failure_reasons:
        typer.echo("Strict comparison failed:")
        for reason in result.strict_failure_reasons:
            typer.echo(f"- {reason}")
        raise typer.Exit(code=1)


@app.command()
def profile(
    model: Annotated[str | None, typer.Option("--model", help="Model to profile.")] = None,
    contexts: Annotated[str | None, typer.Option("--contexts", help="Comma-separated context sizes.")] = None,
    benchmark_types: Annotated[
        str | None,
        typer.Option("--benchmark-types", help="Comma-separated benchmark ids."),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Seed for benchmark requests.")] = 42,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature",
            min=0.0,
            max=2.0,
            help="Sampling temperature for benchmark requests.",
        ),
    ] = 0.0,
    top_p: Annotated[
        float | None,
        typer.Option(
            "--top-p",
            min=0.0,
            max=1.0,
            help="Top-p sampling value for benchmark requests.",
        ),
    ] = None,
    repetitions: Annotated[
        int,
        typer.Option("--repetitions", min=1, help="Number of repeated runs per scenario."),
    ] = 1,
    warmup_runs: Annotated[
        int,
        typer.Option("--warmup-runs", min=1, help="Silent warmup runs per context size."),
    ] = 1,
    no_warmup: Annotated[
        bool,
        typer.Option("--no-warmup", help="Disable silent warmup runs."),
    ] = False,
    live_progress: Annotated[
        bool,
        typer.Option(
            "--live-progress",
            help="Show live benchmark status, telemetry, and per-run summaries.",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            help="Skip the final benchmark session confirmation prompt.",
        ),
    ] = False,
    output_dir: Path = typer.Option(
        Path("results"),
        "--output-dir",
        help="Base directory for timestamped session artifacts.",
    ),
    ) -> None:
    """Run an Ollama workload profile."""
    execution_settings = _build_execution_settings(
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        repetitions=repetitions,
        warmup_runs=warmup_runs,
        warmup_enabled=not no_warmup,
    )

    with OllamaClient() as client:
        available_models = client.list_models()
        if not available_models:
            typer.echo(
                "No local Ollama models are available. "
                "Pull one with `ollama pull <model>` and try again."
            )
            raise typer.Exit(code=1)

        selected_model = model or typer.prompt(
            "Model to profile",
            default=available_models[0],
            show_default=True,
        )
        if selected_model not in available_models:
            typer.echo(
                f"Selected model {selected_model!r} is not available locally. "
                f"Available models: {', '.join(available_models)}"
            )
            raise typer.Exit(code=1)

        raw_contexts = contexts or typer.prompt("Contexts", default="4096", show_default=True)
        raw_benchmark_types = benchmark_types or typer.prompt(
            "Benchmark types",
            default="smoke",
            show_default=True,
        )
        parsed_contexts = parse_contexts(raw_contexts)
        parsed_benchmark_types = parse_benchmark_types(raw_benchmark_types)
        plan = build_profile_session_plan(
            model_name=selected_model,
            contexts=parsed_contexts,
            benchmark_types=parsed_benchmark_types,
            execution_settings=execution_settings,
        )
        expanded_plan = expand_session_plan(plan)
        budget = summarize_session_budget(plan, expanded_plan=expanded_plan)

        typer.echo(f"Model: {selected_model}")
        typer.echo(f"Contexts: {', '.join(str(context) for context in parsed_contexts)}")
        typer.echo(
            "Benchmark types: "
            + ", ".join(benchmark_type.value for benchmark_type in parsed_benchmark_types)
        )
        typer.echo(_format_benchmark_policy(execution_settings))
        typer.echo(
            "Benchmark budget: "
            f"{budget['run_count']} run(s) across {budget['scenario_count']} scenario(s) "
            f"from {budget['context_count']} context selection(s) and "
            f"{budget['benchmark_type_count']} benchmark selection(s) "
            f"with repetitions={budget['repetitions']}."
        )
        if budget["warning"] is not None:
            typer.echo(f"Budget warning: {budget['warning']}")
        if not yes and not typer.confirm("Proceed with this benchmark session?", default=True):
            typer.echo("Profile session cancelled.")
            raise typer.Exit(code=1)

        result = run_profile_session(
            plan=plan,
            client=client,
            output_dir=output_dir,
            available_models=available_models,
            expanded_plan=expanded_plan,
            progress_reporter=TerminalProgressReporter(
                echo=_build_terminal_echo(),
                live=live_progress,
            ),
        )

    typer.echo(f"Benchmark methodology: {BENCHMARK_METHODOLOGY_VERSION}")
    typer.echo(f"Session artifacts written to: {result.session_dir}")


def main() -> None:
    app()
