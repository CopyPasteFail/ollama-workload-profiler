from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import os
import platform
import shutil
import sys
from urllib import error, request


@dataclass(frozen=True)
class PythonEnvironmentStatus:
    python_version: str
    in_venv: bool
    executable: str


@dataclass(frozen=True)
class DoctorSummary:
    exit_code: int
    platform_name: str
    messages: list[str]
    remediation_hints: list[str]


def detect_python_environment() -> PythonEnvironmentStatus:
    in_venv = (
        bool(os.environ.get("VIRTUAL_ENV"))
        or sys.prefix != sys.base_prefix
        or sys.exec_prefix != sys.base_exec_prefix
    )
    return PythonEnvironmentStatus(
        python_version=sys.version.split()[0],
        in_venv=in_venv,
        executable=sys.executable,
    )


def detect_ollama_binary() -> bool:
    return shutil.which("ollama") is not None


def detect_dependency_status() -> tuple[bool, list[str]]:
    required_modules = (
        "typer",
        "rich",
        "httpx",
        "psutil",
        "pydantic",
        "jinja2",
    )
    missing = [
        module_name
        for module_name in required_modules
        if importlib.util.find_spec(module_name) is None
    ]
    return (not missing, missing)


def probe_ollama_server(base_url: str = "http://127.0.0.1:11434") -> tuple[bool, list[str]]:
    url = f"{base_url}/api/tags"
    try:
        with request.urlopen(url, timeout=5) as response:
            payload = json.load(response)
    except (error.URLError, TimeoutError, json.JSONDecodeError):
        return False, []

    models = [
        item["name"]
        for item in payload.get("models", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]
    return True, models


def summarize_doctor_status(
    binary_found: bool,
    reachable: bool | None,
    models: list[str] | None,
    dependencies_ok: bool = True,
    missing_dependencies: list[str] | None = None,
) -> DoctorSummary:
    messages: list[str] = []
    remediation_hints: list[str] = []
    exit_code = 0

    if dependencies_ok:
        messages.append("dependencies: ok")
    else:
        missing_display = ", ".join(missing_dependencies or [])
        messages.append(f"dependencies: missing ({missing_display})")
        remediation_hints.append(
            "Run `python scripts/bootstrap.py` to install the pinned Python dependencies."
        )
        exit_code = 1

    if binary_found:
        messages.append("ollama binary: found")
    else:
        messages.append("ollama binary: missing")
        remediation_hints.append(
            "Install Ollama and ensure the `ollama` command is available on PATH."
        )
        exit_code = 1

    if reachable is None:
        messages.append("ollama server: not checked yet")
        remediation_hints.append(
            "Start the Ollama server and rerun `owp doctor` for a full reachability check."
        )
        exit_code = 1
    elif reachable:
        messages.append("ollama server: reachable")
    else:
        messages.append("ollama server: unreachable")
        remediation_hints.append("Start Ollama locally before running `owp profile`.")
        exit_code = 1

    if models is None:
        messages.append("models: not checked yet")
        remediation_hints.append(
            "Pull a model with `ollama pull <model>` after the server is reachable."
        )
        exit_code = 1
    elif models:
        messages.append(f"models: {', '.join(models)}")
    else:
        messages.append("models: none found")
        remediation_hints.append("Pull a local model with `ollama pull <model>`.")
        exit_code = 1

    return DoctorSummary(
        exit_code=exit_code,
        platform_name=platform.platform(),
        messages=messages,
        remediation_hints=remediation_hints,
    )
