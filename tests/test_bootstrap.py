from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


def load_bootstrap_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap.py"
    spec = importlib.util.spec_from_file_location("bootstrap", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_install_requirements_streams_pip_output(monkeypatch, capsys, tmp_path):
    bootstrap = load_bootstrap_module()
    calls = []

    class FakeProcess:
        def __init__(self):
            self.stdout = iter(["Requirement already satisfied: typer\n"])
            self.returncode = 0

        def wait(self):
            return self.returncode

    def fake_popen(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeProcess()

    monkeypatch.setattr(bootstrap.subprocess, "Popen", fake_popen)

    changed = bootstrap.install_requirements(
        tmp_path / "python.exe",
        tmp_path / "requirements.txt",
    )

    assert changed is False
    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["cwd"] == bootstrap.PROJECT_ROOT
    assert kwargs["text"] is True
    assert kwargs["stdout"] == subprocess.PIPE
    assert kwargs["stderr"] == subprocess.STDOUT

    output = capsys.readouterr().out
    assert "installing dependencies..." in output
    assert "Requirement already satisfied: typer" in output


def test_install_requirements_reports_changes_when_pip_installs(monkeypatch, tmp_path):
    bootstrap = load_bootstrap_module()

    class FakeProcess:
        def __init__(self):
            self.stdout = iter(
                [
                    "Collecting typer\n",
                    "Installing collected packages: typer\n",
                    "Successfully installed typer-0.16.0\n",
                ]
            )
            self.returncode = 0

        def wait(self):
            return self.returncode

    monkeypatch.setattr(bootstrap.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    changed = bootstrap.install_requirements(
        tmp_path / "python.exe",
        tmp_path / "requirements.txt",
    )

    assert changed is True


def test_install_editable_project_streams_pip_output(monkeypatch, capsys, tmp_path):
    bootstrap = load_bootstrap_module()
    calls = []

    class FakeProcess:
        def __init__(self):
            self.stdout = iter(["Requirement already satisfied: ollama-workload-profiler in c:\\repo\n"])
            self.returncode = 0

        def wait(self):
            return self.returncode

    def fake_popen(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeProcess()

    monkeypatch.setattr(bootstrap.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(bootstrap, "is_editable_install_satisfied", lambda _python: False)

    changed = bootstrap.install_editable_project(tmp_path / "python.exe")

    assert changed is False
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert list(args[0][-3:]) == ["install", "-e", "."]
    assert kwargs["cwd"] == bootstrap.PROJECT_ROOT
    assert kwargs["text"] is True
    assert kwargs["stdout"] == subprocess.PIPE
    assert kwargs["stderr"] == subprocess.STDOUT

    output = capsys.readouterr().out
    assert "installing project in editable mode..." in output
    assert "Requirement already satisfied: ollama-workload-profiler" in output


def test_install_editable_project_skips_pip_when_already_satisfied(monkeypatch, capsys, tmp_path):
    bootstrap = load_bootstrap_module()
    popen_calls = []

    monkeypatch.setattr(bootstrap, "is_editable_install_satisfied", lambda _python: True)
    monkeypatch.setattr(bootstrap.subprocess, "Popen", lambda *args, **kwargs: popen_calls.append((args, kwargs)))

    changed = bootstrap.install_editable_project(tmp_path / "python.exe")

    assert changed is False
    assert popen_calls == []
    assert "editable install already satisfied." in capsys.readouterr().out


def test_main_installs_editable_project_after_dependencies(monkeypatch, capsys, tmp_path):
    bootstrap = load_bootstrap_module()
    events: list[tuple[str, object]] = []
    python_path = tmp_path / "Scripts" / "python.exe"

    class FakeEnvCheck:
        @staticmethod
        def detect_ollama_binary() -> bool:
            return True

        @staticmethod
        def probe_ollama_server() -> tuple[bool, list[str]]:
            return True, ["llama3.2"]

    monkeypatch.setattr(
        bootstrap,
        "ensure_virtualenv",
        lambda _path: events.append(("ensure_virtualenv", _path)) or False,
    )
    monkeypatch.setattr(
        bootstrap,
        "find_venv_python",
        lambda _path: events.append(("find_venv_python", _path)) or python_path,
    )
    monkeypatch.setattr(
        bootstrap,
        "install_requirements",
        lambda python, requirements: events.append(("install_requirements", (python, requirements))) or False,
    )
    monkeypatch.setattr(
        bootstrap,
        "install_editable_project",
        lambda python: events.append(("install_editable_project", python)) or False,
    )
    monkeypatch.setattr(bootstrap, "load_env_check_module", lambda: FakeEnvCheck)

    exit_code = bootstrap.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert events == [
        ("ensure_virtualenv", bootstrap.VENV_PATH),
        ("find_venv_python", bootstrap.VENV_PATH),
        ("install_requirements", (python_path, bootstrap.REQUIREMENTS_PATH)),
        ("install_editable_project", python_path),
    ]
    assert "package already installed in editable mode" in captured.out
