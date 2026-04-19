from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import os
import platform
import shutil
import subprocess
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


def detect_accelerator_metadata() -> dict[str, object]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        metadata = _detect_nvidia_metadata(nvidia_smi)
        if metadata is not None:
            return metadata

    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi is not None:
        metadata = _detect_amd_metadata(rocm_smi)
        if metadata is not None:
            return metadata

    system_name = _safe_platform_value(platform.system)
    machine = _safe_platform_value(platform.machine)
    if system_name == "Darwin" and isinstance(machine, str) and machine.startswith(("arm", "ARM")):
        return {
            "kind": "apple_silicon",
            "detection_source": "platform",
            "available": True,
            "device_count": 1,
            "devices": [{"name": "Apple Silicon integrated GPU", "memory_total_mb": None}],
            "status": "detected",
            "notes": ["Detected Apple Silicon host; integrated accelerator details are not exposed via a local CLI probe."],
        }

    return {
        "kind": "unknown",
        "detection_source": "heuristic",
        "available": False,
        "device_count": 0,
        "devices": [],
        "status": "undetected",
        "notes": ["No supported accelerator tooling detected; host is likely CPU-only."],
    }


def _detect_nvidia_metadata(command: str) -> dict[str, object] | None:
    try:
        result = subprocess.run(
            [
                command,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return {
            "kind": "nvidia",
            "detection_source": "nvidia-smi",
            "available": False,
            "device_count": 0,
            "devices": [],
            "status": "unavailable",
            "notes": ["`nvidia-smi` is present but did not return GPU details."],
        }

    devices: list[dict[str, object]] = []
    driver_version: str | None = None
    for raw_line in result.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        memory_total_mb = _parse_int(parts[1])
        driver_version = driver_version or (parts[2] or None)
        devices.append(
            {
                "name": parts[0],
                "memory_total_mb": memory_total_mb,
            }
        )

    if not devices:
        return None

    return {
        "kind": "nvidia",
        "detection_source": "nvidia-smi",
        "available": True,
        "device_count": len(devices),
        "devices": devices,
        "driver_version": driver_version,
        "status": "detected",
        "notes": [],
    }


def _detect_amd_metadata(command: str) -> dict[str, object] | None:
    try:
        result = subprocess.run(
            [command, "--showproductname", "--showmeminfo", "vram"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return {
            "kind": "amd",
            "detection_source": "rocm-smi",
            "available": False,
            "device_count": 0,
            "devices": [],
            "status": "unavailable",
            "notes": ["`rocm-smi` is present but did not return GPU details."],
        }

    devices: list[dict[str, object]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line or "GPU[" not in line:
            continue
        if "Card series:" in line:
            name = line.split("Card series:", 1)[1].strip() or None
            devices.append({"name": name, "memory_total_mb": None})
            continue
        if "Total Memory (B):" in line and devices:
            raw_value = line.split("Total Memory (B):", 1)[1].strip()
            memory_bytes = _parse_int(raw_value)
            if memory_bytes is not None:
                devices[-1]["memory_total_mb"] = round(memory_bytes / (1024 * 1024), 3)

    if not devices:
        return None

    return {
        "kind": "amd",
        "detection_source": "rocm-smi",
        "available": True,
        "device_count": len(devices),
        "devices": devices,
        "status": "detected",
        "notes": [],
    }


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_platform_value(factory: object) -> str | None:
    if not callable(factory):
        return None
    try:
        value = factory()
    except Exception:
        return None
    if not isinstance(value, str):
        return None
    return value or None


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
