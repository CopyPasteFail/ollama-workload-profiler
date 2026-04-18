import json
from pathlib import Path

import pytest
from typer import BadParameter
from typer.testing import CliRunner

from ollama_workload_profiler.cli import app, parse_benchmark_types, parse_contexts
from ollama_workload_profiler.models.plan import BenchmarkType


def test_cli_shows_root_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "doctor" in result.stdout
    assert "profile" in result.stdout


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
