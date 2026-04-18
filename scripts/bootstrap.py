from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ollama_workload_profiler.env_check import detect_ollama_binary, probe_ollama_server


VENV_PATH = PROJECT_ROOT / ".venv"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"


@dataclass(frozen=True)
class BootstrapStatus:
    messages: list[str] = field(default_factory=list)
    exit_code: int = 0


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


def install_requirements(python_executable: Path, requirements_path: Path) -> bool:
    result = subprocess.run(
        [
            str(python_executable),
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_path),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    output = "\n".join(
        chunk for chunk in (result.stdout, result.stderr) if chunk.strip()
    )
    normalized_output = output.lower()
    return "already satisfied" not in normalized_output


def summarize_bootstrap_status(
    *,
    venv_created: bool,
    deps_installed: bool,
    ollama_binary_found: bool,
    ollama_reachable: bool,
    models: list[str],
) -> BootstrapStatus:
    messages = [
        "venv created" if venv_created else "venv reused",
        "deps installed" if deps_installed else "deps already satisfied",
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
    venv_created = ensure_virtualenv(VENV_PATH)
    python_executable = find_venv_python(VENV_PATH)
    deps_installed = install_requirements(python_executable, REQUIREMENTS_PATH)
    ollama_binary_found = detect_ollama_binary()
    ollama_reachable, models = probe_ollama_server()
    status = summarize_bootstrap_status(
        venv_created=venv_created,
        deps_installed=deps_installed,
        ollama_binary_found=ollama_binary_found,
        ollama_reachable=ollama_reachable,
        models=models,
    )

    for message in status.messages:
        print(message)

    return status.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
