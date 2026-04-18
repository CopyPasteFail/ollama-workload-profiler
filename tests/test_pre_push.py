from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def load_pre_push_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_pre_push_checks.py"
    spec = importlib.util.spec_from_file_location("run_pre_push_checks", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pre_push_runner_executes_expected_checks_in_order(monkeypatch, capsys) -> None:
    runner = load_pre_push_module()
    commands: list[list[str]] = []

    class Completed:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    def fake_run(command: list[str], cwd: Path) -> Completed:
        commands.append(command)
        assert cwd == runner.PROJECT_ROOT
        return Completed(0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    exit_code = runner.main()

    assert exit_code == 0
    assert commands == [
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "compileall", "-q", "src", "tests"],
        [sys.executable, "-m", "pytest", "-q"],
    ]
    output = capsys.readouterr().out
    assert "[pre-push] Ruff lint..." in output
    assert "[pre-push] Compile check..." in output
    assert "[pre-push] Pytest..." in output
    assert "[pre-push] All checks passed." in output


def test_pre_push_runner_stops_after_first_failure(monkeypatch, capsys) -> None:
    runner = load_pre_push_module()
    commands: list[list[str]] = []

    class Completed:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    responses = iter([Completed(1), Completed(0)])

    def fake_run(command: list[str], cwd: Path) -> Completed:
        commands.append(command)
        assert cwd == runner.PROJECT_ROOT
        return next(responses)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    exit_code = runner.main()

    assert exit_code == 1
    assert commands == [[sys.executable, "-m", "ruff", "check", "."]]
    assert "Ruff lint failed with exit code 1." in capsys.readouterr().err


def test_pre_push_hook_invokes_repo_runner() -> None:
    hook_path = Path(__file__).resolve().parents[1] / ".githooks" / "pre-push"
    content = hook_path.read_text(encoding="utf-8")

    assert 'ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"' in content
    assert 'exec "$PYTHON_BIN" "$ROOT/scripts/run_pre_push_checks.py"' in content


def test_pinned_dependencies_cover_pre_push_python_modules() -> None:
    project_root = Path(__file__).resolve().parents[1]
    requirements = (project_root / "requirements.txt").read_text(encoding="utf-8")
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")

    assert "ruff==" in requirements
    assert "pytest==" in requirements
    assert "ruff==" in pyproject
    assert "pytest==" in pyproject
