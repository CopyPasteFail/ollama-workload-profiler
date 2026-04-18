from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PATH = PROJECT_ROOT / ".venv"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"


@dataclass(frozen=True)
class BootstrapStatus:
    messages: list[str] = field(default_factory=list)
    exit_code: int = 0


def load_env_check_module():
    module_path = PROJECT_ROOT / "src" / "ollama_workload_profiler" / "env_check.py"
    module_name = "bootstrap_env_check"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load env_check module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def create_virtualenv(venv_path: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_path)],
        check=True,
        cwd=PROJECT_ROOT,
    )


def ensure_virtualenv(venv_path: Path) -> bool:
    if venv_path.exists():
        return False

    create_virtualenv(venv_path)
    return True


def find_venv_python(venv_path: Path) -> Path:
    candidates = (
        venv_path / "Scripts" / "python.exe",
        venv_path / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def run_pip_install(command: list[str], status_message: str) -> bool:
    print(status_message)
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, process.args)

    normalized_output = "\n".join(output_lines).lower()
    installed = (
        "successfully installed" in normalized_output
        or "installing collected packages" in normalized_output
        or "building editable for" in normalized_output
        or "running setup.py develop" in normalized_output
    )
    already_satisfied = "requirement already satisfied" in normalized_output
    return installed or not already_satisfied


def install_requirements(python_executable: Path, requirements_path: Path) -> bool:
    return run_pip_install(
        [
            str(python_executable),
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_path),
        ],
        "installing dependencies...",
    )


def is_editable_install_satisfied(python_executable: Path) -> bool:
    process = subprocess.run(
        [
            str(python_executable),
            "-c",
            (
                "from pathlib import Path; "
                "import ollama_workload_profiler as package; "
                "print(Path(package.__file__).resolve())"
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        return False

    installed_path = Path(process.stdout.strip())
    expected_path = (PROJECT_ROOT / "src" / "ollama_workload_profiler" / "__init__.py").resolve()
    return installed_path == expected_path


def install_editable_project(python_executable: Path) -> bool:
    if is_editable_install_satisfied(python_executable):
        print("editable install already satisfied.")
        return False

    return run_pip_install(
        [
            str(python_executable),
            "-m",
            "pip",
            "install",
            "-e",
            ".",
        ],
        "installing project in editable mode...",
    )


def summarize_bootstrap_status(
    *,
    venv_created: bool,
    deps_installed: bool,
    package_installed: bool,
    ollama_binary_found: bool,
    ollama_reachable: bool,
    models: list[str],
) -> BootstrapStatus:
    messages = [
        "venv created" if venv_created else "venv reused",
        "deps installed" if deps_installed else "deps already satisfied",
        (
            "package installed in editable mode"
            if package_installed
            else "package already installed in editable mode"
        ),
    ]
    exit_code = 0

    if ollama_binary_found:
        messages.append("ollama binary: found")
    else:
        messages.append("ollama binary: missing")
        exit_code = 1

    if ollama_reachable:
        messages.append("ollama server: reachable")
    else:
        messages.append("ollama server: unreachable")
        exit_code = 1

    if models:
        messages.append(f"models: {', '.join(models)}")
    else:
        messages.append("models: none found")
        exit_code = 1

    return BootstrapStatus(messages=messages, exit_code=exit_code)


def main() -> int:
    env_check = load_env_check_module()
    venv_created = ensure_virtualenv(VENV_PATH)
    python_executable = find_venv_python(VENV_PATH)
    deps_installed = install_requirements(python_executable, REQUIREMENTS_PATH)
    package_installed = install_editable_project(python_executable)
    ollama_binary_found = env_check.detect_ollama_binary()
    ollama_reachable, models = env_check.probe_ollama_server()
    status = summarize_bootstrap_status(
        venv_created=venv_created,
        deps_installed=deps_installed,
        package_installed=package_installed,
        ollama_binary_found=ollama_binary_found,
        ollama_reachable=ollama_reachable,
        models=models,
    )

    for message in status.messages:
        print(message)

    return status.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
