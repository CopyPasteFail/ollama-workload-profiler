import json
from pathlib import Path
import sys

import pytest
from typer import BadParameter
from typer.testing import CliRunner

from ollama_workload_profiler.methodology import BENCHMARK_METHODOLOGY_VERSION
from ollama_workload_profiler.cli import (
    _build_terminal_echo,
    app,
    parse_benchmark_types,
    parse_contexts,
)
from ollama_workload_profiler.models.plan import BenchmarkType


def test_cli_shows_root_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "doctor" in result.stdout
    assert "profile" in result.stdout


def test_profile_help_includes_live_progress_flag() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["profile", "--help"])

    assert result.exit_code == 0
    assert "--live-progress" in result.stdout


def test_profile_help_includes_benchmark_policy_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["profile", "--help"])

    assert result.exit_code == 0
    assert "--seed" in result.stdout
    assert "--temperature" in result.stdout
    assert "--top-p" in result.stdout
    assert "--repetitions" in result.stdout
    assert "--warmup-runs" in result.stdout
    assert "--no-warmup" in result.stdout


def _write_compare_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_compare_session(path: Path, *, temperature: float = 0.0) -> None:
    path.mkdir()
    execution_settings = {
        "seed": 42,
        "temperature": temperature,
        "top_p": None,
        "repetitions": 1,
        "warmup_runs": 1,
        "warmup_enabled": True,
    }
    _write_compare_json(
        path / "summary.json",
        {
            "benchmark_methodology_version": BENCHMARK_METHODOLOGY_VERSION,
            "session_metrics": {
                "run_count": 1,
                "completed_runs": 1,
                "failed_runs": 0,
                "stopped_runs": 0,
                "completed_sample_size": 1,
                "elapsed_ms_median": 100.0,
                "elapsed_ms_p95": 100.0,
                "elapsed_ms_sample_size": 1,
            },
            "benchmark_summaries": [
                {
                    "model_name": "llama3.2",
                    "context_size": 4096,
                    "benchmark_type": "smoke",
                    "scenario_id": "smoke-basic-v1",
                    "strict_sample_size": 1,
                    "elapsed_ms_median": 100.0,
                    "elapsed_ms_p95": 100.0,
                    "elapsed_ms_sample_size": 1,
                }
            ],
        },
    )
    _write_compare_json(
        path / "plan.json",
        {
            "benchmark_methodology_version": BENCHMARK_METHODOLOGY_VERSION,
            "model_name": "llama3.2",
            "contexts": [4096],
            "benchmark_types": ["smoke"],
            "execution_settings": execution_settings,
        },
    )
    _write_compare_json(
        path / "environment.json",
        {
            "benchmark_methodology_version": BENCHMARK_METHODOLOGY_VERSION,
            "execution_settings": execution_settings,
        },
    )


def test_compare_help_is_available() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["compare", "--help"])

    assert result.exit_code == 0
    assert "--format" in result.stdout
    assert "--strict" in result.stdout
    assert "--output" in result.stdout
    assert "--all-metrics" in result.stdout


def test_compare_strict_returns_nonzero_for_strict_blocking_warnings(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_compare_session(baseline, temperature=0.0)
    _write_compare_session(candidate, temperature=0.2)

    runner = CliRunner()
    result = runner.invoke(app, ["compare", str(baseline), str(candidate), "--strict"])

    assert result.exit_code == 1
    assert "Strict comparison failed:" in result.stdout
    assert "execution_policy.temperature_mismatch" in result.stdout


def test_compare_writes_selected_format_to_output_and_prints_completion_line(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    output_path = tmp_path / "compare.json"
    _write_compare_session(baseline)
    _write_compare_session(candidate)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "compare",
            str(baseline),
            str(candidate),
            "--format",
            "json",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert f"Comparison written to: {output_path}" in result.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["comparability_status"] == "pass"


def test_compare_output_creates_missing_parent_directories(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    output_path = tmp_path / "nested" / "reports" / "compare.txt"
    _write_compare_session(baseline)
    _write_compare_session(candidate)

    runner = CliRunner()
    result = runner.invoke(app, ["compare", str(baseline), str(candidate), "--output", str(output_path)])

    assert result.exit_code == 0
    assert output_path.exists()
    assert f"Comparison written to: {output_path}" in result.stdout


def test_compare_default_output_includes_human_sections(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_compare_session(baseline)
    _write_compare_session(candidate)

    runner = CliRunner()
    result = runner.invoke(app, ["compare", str(baseline), str(candidate)])

    assert result.exit_code == 0
    assert "Comparison: baseline -> candidate" in result.stdout
    assert "Warnings:" in result.stdout
    assert "Session Summary:" in result.stdout
    assert "Rows:" in result.stdout
    assert "Unmatched Rows:" in result.stdout


def test_compare_all_metrics_shows_unavailable_and_unchanged_metrics(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_compare_session(baseline)
    _write_compare_session(candidate)

    runner = CliRunner()
    default_result = runner.invoke(app, ["compare", str(baseline), str(candidate)])
    all_metrics_result = runner.invoke(app, ["compare", str(baseline), str(candidate), "--all-metrics"])

    assert default_result.exit_code == 0
    assert all_metrics_result.exit_code == 0
    assert "No changed metrics." in default_result.stdout
    assert "elapsed_ms_median" not in default_result.stdout
    assert "elapsed_ms_median" in all_metrics_result.stdout


def test_profile_rejects_out_of_range_temperature_before_plan_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_build_profile_session_plan(*args: object, **kwargs: object) -> object:
        raise AssertionError("plan construction should not run for invalid temperature")

    monkeypatch.setattr("ollama_workload_profiler.cli.build_profile_session_plan", fake_build_profile_session_plan)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "--model",
            "llama3.2",
            "--contexts",
            "4096",
            "--benchmark-types",
            "smoke",
            "--temperature",
            "-0.1",
        ],
    )

    assert result.exit_code == 2


def test_profile_rejects_out_of_range_top_p_before_plan_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_build_profile_session_plan(*args: object, **kwargs: object) -> object:
        raise AssertionError("plan construction should not run for invalid top_p")

    monkeypatch.setattr("ollama_workload_profiler.cli.build_profile_session_plan", fake_build_profile_session_plan)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "--model",
            "llama3.2",
            "--contexts",
            "4096",
            "--benchmark-types",
            "smoke",
            "--top-p",
            "1.1",
        ],
    )

    assert result.exit_code == 2


def test_build_terminal_echo_writes_directly_to_stdout_and_flushes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[str] = []
    flush_calls = 0

    class FakeStdout:
        def write(self, message: str) -> int:
            writes.append(message)
            return len(message)

        def flush(self) -> None:
            nonlocal flush_calls
            flush_calls += 1

    monkeypatch.setattr(sys, "stdout", FakeStdout())

    echo = _build_terminal_echo()
    echo("\rLive | example", nl=False)
    echo("Result | example", nl=True)

    assert writes == ["\rLive | example", "Result | example\n"]
    assert flush_calls == 2


def test_cli_module_entrypoint_runs() -> None:
    from ollama_workload_profiler.__main__ import main

    assert callable(main)


def test_parse_contexts_preserves_order_and_deduplicates() -> None:
    assert parse_contexts(" 8, 16, 8, 32 ") == [8, 16, 32]


def test_parse_contexts_rejects_empty_selection() -> None:
    with pytest.raises(BadParameter, match="Select at least one context."):
        parse_contexts("   ")


def test_parse_contexts_rejects_garbage_tokens() -> None:
    with pytest.raises(BadParameter, match="Contexts must be comma-separated integers."):
        parse_contexts("8, sixteen")


def test_parse_contexts_rejects_negative_values() -> None:
    with pytest.raises(BadParameter, match="Contexts must be positive integers."):
        parse_contexts("-1")


def test_parse_contexts_rejects_non_positive_values() -> None:
    with pytest.raises(BadParameter, match="Contexts must be positive integers."):
        parse_contexts("0")


def test_parse_benchmark_types_preserves_order_and_deduplicates() -> None:
    assert parse_benchmark_types("smoke, context-scaling, smoke") == [
        BenchmarkType.SMOKE,
        BenchmarkType.CONTEXT_SCALING,
    ]


def test_parse_benchmark_types_accepts_all_supported_benchmark_families() -> None:
    assert parse_benchmark_types("smoke, cold-warm, output-scaling, ttft, stress") == [
        BenchmarkType.SMOKE,
        BenchmarkType.COLD_WARM,
        BenchmarkType.OUTPUT_SCALING,
        BenchmarkType.TTFT,
        BenchmarkType.STRESS,
    ]


def test_parse_benchmark_types_rejects_empty_tokens() -> None:
    with pytest.raises(BadParameter, match="Benchmark types must not contain empty values."):
        parse_benchmark_types("smoke,")


def test_parse_benchmark_types_rejects_unknown_ids() -> None:
    with pytest.raises(BadParameter, match="Invalid benchmark type: unknown."):
        parse_benchmark_types("smoke, unknown")


def test_profile_command_parses_and_echoes_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeClient:
        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def list_models(self) -> list[str]:
            return ["llama3.2"]

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            return {
                "response": "ok",
                "total_duration": 1_000_000,
                "eval_count": 4,
                "eval_duration": 250_000,
                "prompt_eval_count": 8,
                "prompt_eval_duration": 500_000,
            }

    monkeypatch.setattr("ollama_workload_profiler.cli.OllamaClient", FakeClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "--model",
            "llama3.2",
            "--contexts",
            "8192, 4096, 8192",
            "--benchmark-types",
            "context-scaling, smoke",
            "--output-dir",
            str(tmp_path),
        ],
        input="y\n",
    )

    assert result.exit_code == 0
    assert "Model: llama3.2" in result.stdout
    assert "Contexts: 8192, 4096" in result.stdout
    assert "Benchmark types: context-scaling, smoke" in result.stdout
    assert f"Benchmark methodology: {BENCHMARK_METHODOLOGY_VERSION}" in result.stdout


def test_profile_command_passes_requested_execution_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_settings: dict[str, object] = {}

    class FakeClient:
        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def list_models(self) -> list[str]:
            return ["llama3.2"]

    def fake_build_profile_session_plan(
        *,
        model_name: str,
        contexts: list[int],
        benchmark_types: list[BenchmarkType],
        execution_settings: dict[str, object],
    ) -> object:
        captured_settings.update(execution_settings)
        return type(
            "FakePlan",
            (),
            {
                "model_name": model_name,
                "contexts": contexts,
                "benchmark_types": benchmark_types,
                "execution_settings": execution_settings,
            },
        )()

    def fake_expand_session_plan(plan: object) -> list[object]:
        return []

    def fake_summarize_session_budget(plan: object, *, expanded_plan: list[object]) -> dict[str, object]:
        return {
            "run_count": 0,
            "scenario_count": 0,
            "context_count": 1,
            "benchmark_type_count": 1,
            "repetitions": 1,
            "warning": None,
        }

    def fake_run_profile_session(
        *,
        plan: object,
        client: object,
        output_dir: Path,
        available_models: list[str],
        expanded_plan: list[object],
        progress_reporter: object,
    ) -> object:
        return type("FakeResult", (), {"session_dir": tmp_path / "session"})()

    monkeypatch.setattr("ollama_workload_profiler.cli.OllamaClient", FakeClient)
    monkeypatch.setattr("ollama_workload_profiler.cli.build_profile_session_plan", fake_build_profile_session_plan)
    monkeypatch.setattr("ollama_workload_profiler.cli.expand_session_plan", fake_expand_session_plan)
    monkeypatch.setattr("ollama_workload_profiler.cli.summarize_session_budget", fake_summarize_session_budget)
    monkeypatch.setattr("ollama_workload_profiler.cli.run_profile_session", fake_run_profile_session)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "--model",
            "llama3.2",
            "--contexts",
            "4096",
            "--benchmark-types",
            "smoke",
            "--seed",
            "7",
            "--temperature",
            "0.2",
            "--top-p",
            "0.9",
            "--repetitions",
            "3",
            "--warmup-runs",
            "2",
            "--no-warmup",
            "--output-dir",
            str(tmp_path),
        ],
        input="y\n",
    )

    assert result.exit_code == 0
    assert captured_settings == {
        "seed": 7,
        "temperature": 0.2,
        "top_p": 0.9,
        "repetitions": 3,
        "warmup_runs": 2,
        "warmup_enabled": False,
    }
    assert "Benchmark policy:" in result.stdout
    assert "seed=7" in result.stdout
    assert "temperature=0.2" in result.stdout
    assert "top_p=0.9" in result.stdout
    assert "repetitions=3" in result.stdout
    assert "warmup_runs=2" in result.stdout
    assert "warmup_enabled=no" in result.stdout


def test_profile_refuses_to_start_when_no_local_models_are_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def list_models(self) -> list[str]:
            return []

    monkeypatch.setattr("ollama_workload_profiler.cli.OllamaClient", FakeClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "--model",
            "llama3.2",
            "--contexts",
            "4096",
            "--benchmark-types",
            "smoke",
        ],
    )

    assert result.exit_code != 0
    assert "No local Ollama models" in result.stdout


def test_profile_runs_interactively_when_options_are_omitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeClient:
        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def list_models(self) -> list[str]:
            return ["llama3.2"]

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            return {
                "response": "ok",
                "total_duration": 1_000_000,
                "eval_count": 4,
                "eval_duration": 250_000,
                "prompt_eval_count": 8,
                "prompt_eval_duration": 500_000,
            }

    monkeypatch.setattr("ollama_workload_profiler.cli.OllamaClient", FakeClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["profile", "--output-dir", str(tmp_path)],
        input="llama3.2\n4096\nsmoke\ny\n",
    )

    assert result.exit_code == 0
    assert "Model to profile" in result.stdout
    assert "Contexts" in result.stdout
    assert "Benchmark types" in result.stdout
    assert "Benchmark budget:" in result.stdout
    assert "Proceed with this benchmark session?" in result.stdout
    assert "Session artifacts written to:" in result.stdout


def test_profile_interactive_flow_allows_user_to_cancel_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeClient:
        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def list_models(self) -> list[str]:
            return ["llama3.2"]

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            raise AssertionError("Execution should not start when the user cancels")

    monkeypatch.setattr("ollama_workload_profiler.cli.OllamaClient", FakeClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["profile", "--output-dir", str(tmp_path)],
        input="llama3.2\n4096\nsmoke\nn\n",
    )

    assert result.exit_code == 1
    assert "Proceed with this benchmark session?" in result.stdout
    assert "Profile session cancelled." in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_profile_yes_flag_skips_confirmation_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeClient:
        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def list_models(self) -> list[str]:
            return ["llama3.2"]

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            return {
                "response": "ok",
                "total_duration": 1_000_000,
                "eval_count": 4,
                "eval_duration": 250_000,
                "prompt_eval_count": 8,
                "prompt_eval_duration": 500_000,
            }

    monkeypatch.setattr("ollama_workload_profiler.cli.OllamaClient", FakeClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "--model",
            "llama3.2",
            "--contexts",
            "4096",
            "--benchmark-types",
            "smoke",
            "--yes",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Proceed with this benchmark session?" not in result.stdout
    assert "Session artifacts written to:" in result.stdout


def test_profile_writes_session_artifacts_on_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.generate_calls = 0

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def list_models(self) -> list[str]:
            return ["llama3.2"]

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            self.generate_calls += 1
            return {
                "response": "ok",
                "total_duration": 1_000_000,
                "eval_count": 4,
                "eval_duration": 250_000,
                "prompt_eval_count": 8,
                "prompt_eval_duration": 500_000,
            }

    monkeypatch.setattr("ollama_workload_profiler.cli.OllamaClient", FakeClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "--model",
            "llama3.2",
            "--contexts",
            "4096",
            "--benchmark-types",
            "smoke",
            "--output-dir",
            str(tmp_path),
        ],
        input="y\n",
    )

    assert result.exit_code == 0
    assert "Model: llama3.2" in result.stdout
    assert "Benchmark budget:" in result.stdout
    assert "repetitions=1" in result.stdout
    assert "Session artifacts written to:" in result.stdout

    session_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(session_dirs) == 1
    session_dir = session_dirs[0]
    assert (session_dir / "plan.json").exists()
    assert (session_dir / "expanded_plan.json").exists()
    assert (session_dir / "environment.json").exists()
    assert (session_dir / "raw.jsonl").exists()
    assert (session_dir / "raw.csv").exists()
    assert (session_dir / "summary.json").exists()
    assert (session_dir / "report.md").exists()

    summary_payload = json.loads((session_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["session_metrics"]["completed_runs"] == 1
    assert summary_payload["session_metrics"]["failed_runs"] == 0
    assert summary_payload["artifacts"]["report.md"] == "report.md"

    expanded_plan_payload = json.loads((session_dir / "expanded_plan.json").read_text(encoding="utf-8"))
    assert [run["benchmark_type"] for run in expanded_plan_payload] == ["smoke"]


def test_profile_persists_artifacts_when_a_run_fails_nonfatally(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.generate_calls = 0

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def list_models(self) -> list[str]:
            return ["llama3.2"]

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            self.generate_calls += 1
            if self.generate_calls == 2:
                raise RuntimeError("simulated generation failure")
            return {
                "response": "ok",
                "total_duration": 1_000_000,
                "eval_count": 4,
                "eval_duration": 250_000,
                "prompt_eval_count": 8,
                "prompt_eval_duration": 500_000,
            }

    monkeypatch.setattr("ollama_workload_profiler.cli.OllamaClient", FakeClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "--model",
            "llama3.2",
            "--contexts",
            "4096,8192",
            "--benchmark-types",
            "smoke",
            "--output-dir",
            str(tmp_path),
        ],
        input="y\n",
    )

    assert result.exit_code == 0
    session_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(session_dirs) == 1
    session_dir = session_dirs[0]

    raw_rows = (session_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw_rows) == 2
    raw_payloads = [json.loads(row) for row in raw_rows]
    assert {row["state"] for row in raw_payloads} == {"completed", "failed"}

    summary_payload = json.loads((session_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["session_metrics"]["completed_runs"] == 1
    assert summary_payload["session_metrics"]["failed_runs"] == 1
    assert "One or more runs failed" in (session_dir / "report.md").read_text(encoding="utf-8")


def test_profile_execution_uses_fixed_benchmark_order_and_warns_for_large_plans(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeClient:
        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def list_models(self) -> list[str]:
            return ["llama3.2"]

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            return {
                "response": "ok",
                "total_duration": 1_000_000,
                "eval_count": 4,
                "eval_duration": 250_000,
                "prompt_eval_count": 8,
                "prompt_eval_duration": 500_000,
            }

    monkeypatch.setattr("ollama_workload_profiler.cli.OllamaClient", FakeClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "--model",
            "llama3.2",
            "--contexts",
            "4096,8192",
            "--benchmark-types",
            "context-scaling,smoke,prompt-scaling,use-case-profiles",
            "--output-dir",
            str(tmp_path),
        ],
        input="y\n",
    )

    assert result.exit_code == 0
    assert "Budget warning:" in result.stdout

    session_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    expanded_plan_payload = json.loads((session_dir / "expanded_plan.json").read_text(encoding="utf-8"))
    benchmark_types = [run["benchmark_type"] for run in expanded_plan_payload]
    assert benchmark_types[:4] == [
        "smoke",
        "prompt-scaling",
        "prompt-scaling",
        "prompt-scaling",
    ]
