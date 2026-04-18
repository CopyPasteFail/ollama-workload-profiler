from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from typer.testing import CliRunner

from ollama_workload_profiler import cli
from ollama_workload_profiler.env_check import (
    PythonEnvironmentStatus,
    detect_dependency_status,
    detect_python_environment,
    probe_ollama_server,
    summarize_doctor_status,
)


def _load_bootstrap_module():
    bootstrap_path = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap.py"
    module_name = "test_bootstrap_module"
    spec = spec_from_file_location(module_name, bootstrap_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_detect_python_environment_reports_venv(monkeypatch) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", r"C:\tmp\venv")
    monkeypatch.setattr("sys.prefix", "C:\\tmp\\venv")
    monkeypatch.setattr("sys.base_prefix", "C:\\Python311")

    status = detect_python_environment()

    assert status.in_venv is True


def test_summarize_doctor_status_returns_nonzero_for_unknown_ollama_state() -> None:
    summary = summarize_doctor_status(
        binary_found=True,
        reachable=None,
        models=None,
    )

    assert summary.exit_code != 0
    assert "dependencies: ok" in summary.messages
    assert "ollama server: not checked yet" in summary.messages
    assert "models: not checked yet" in summary.messages
    assert summary.remediation_hints


def test_detect_dependency_status_reports_missing_modules(monkeypatch) -> None:
    def fake_find_spec(name: str):
        return None if name == "rich" else object()

    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)

    status_ok, missing = detect_dependency_status()

    assert status_ok is False
    assert missing == ["rich"]


def test_probe_ollama_server_returns_models(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, *_args: object, **_kwargs: object) -> bytes:
            return b'{"models": [{"name": "llama3.2:latest"}]}'

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: FakeResponse())

    reachable, models = probe_ollama_server("http://127.0.0.1:11434")

    assert reachable is True
    assert models == ["llama3.2:latest"]


def test_doctor_command_prints_status_and_uses_exit_code(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "detect_python_environment",
        lambda: PythonEnvironmentStatus(
            python_version="3.13.0",
            in_venv=True,
            executable=r"C:\Python313\python.exe",
        ),
    )
    monkeypatch.setattr(
        cli,
        "detect_dependency_status",
        lambda: (True, []),
    )
    monkeypatch.setattr(
        cli,
        "detect_ollama_binary",
        lambda: True,
    )
    monkeypatch.setattr(
        cli,
        "probe_ollama_server",
        lambda: (True, ["llama3.2"]),
    )

    runner = CliRunner()
    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 0
    assert "Platform:" in result.stdout
    assert "Python: 3.13.0" in result.stdout
    assert "dependencies: ok" in result.stdout
    assert "ollama binary: found" in result.stdout
    assert "ollama server: reachable" in result.stdout
    assert "models: llama3.2" in result.stdout
    assert "Remediation hints:" not in result.stdout


def test_bootstrap_reuses_existing_venv(tmp_path, monkeypatch) -> None:
    bootstrap = _load_bootstrap_module()
    venv_path = tmp_path / ".venv"
    venv_path.mkdir()
    created: list[object] = []

    def fake_create_virtualenv(path) -> None:
        created.append(path)

    monkeypatch.setattr(bootstrap, "create_virtualenv", fake_create_virtualenv)

    assert bootstrap.ensure_virtualenv(venv_path) is False
    assert created == []


def test_bootstrap_main_reports_explicit_rerun_status(monkeypatch, capsys) -> None:
    bootstrap = _load_bootstrap_module()

    class FakeEnvCheck:
        @staticmethod
        def detect_ollama_binary() -> bool:
            return True

        @staticmethod
        def probe_ollama_server() -> tuple[bool, list[str]]:
            return True, ["llama3.2"]

    monkeypatch.setattr(bootstrap, "ensure_virtualenv", lambda _path: False)
    monkeypatch.setattr(bootstrap, "find_venv_python", lambda _path: _path / "Scripts" / "python.exe")
    monkeypatch.setattr(bootstrap, "install_requirements", lambda _python, _requirements: False)
    monkeypatch.setattr(bootstrap, "install_editable_project", lambda _python: False)
    monkeypatch.setattr(bootstrap, "load_env_check_module", lambda: FakeEnvCheck)

    exit_code = bootstrap.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "venv reused" in captured.out
    assert "deps already satisfied" in captured.out
    assert "package already installed in editable mode" in captured.out
    assert "ollama binary: found" in captured.out
    assert "ollama server: reachable" in captured.out
    assert "models: llama3.2" in captured.out


def test_find_venv_python_prefers_windows_layout(tmp_path) -> None:
    bootstrap = _load_bootstrap_module()
    venv_path = tmp_path / ".venv"
    scripts_dir = venv_path / "Scripts"
    scripts_dir.mkdir(parents=True)
    python_path = scripts_dir / "python.exe"
    python_path.write_text("", encoding="utf-8")

    assert bootstrap.find_venv_python(venv_path) == python_path


def test_find_venv_python_falls_back_to_posix_layout(tmp_path) -> None:
    bootstrap = _load_bootstrap_module()
    venv_path = tmp_path / ".venv"
    bin_dir = venv_path / "bin"
    bin_dir.mkdir(parents=True)
    python_path = bin_dir / "python"
    python_path.write_text("", encoding="utf-8")

    assert bootstrap.find_venv_python(venv_path) == python_path


def test_summarize_bootstrap_status_distinguishes_rerun_states() -> None:
    bootstrap = _load_bootstrap_module()
    status = bootstrap.summarize_bootstrap_status(
        venv_created=False,
        deps_installed=False,
        package_installed=False,
        ollama_binary_found=False,
        ollama_reachable=False,
        models=[],
    )

    assert status == bootstrap.BootstrapStatus(
        messages=[
            "venv reused",
            "deps already satisfied",
            "package already installed in editable mode",
            "ollama binary: missing",
            "ollama server: unreachable",
            "models: none found",
        ],
        exit_code=1,
    )
