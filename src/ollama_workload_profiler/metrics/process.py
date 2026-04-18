from __future__ import annotations

from collections.abc import Iterable

import psutil


def find_ollama_processes() -> list[psutil.Process]:
    return [proc for proc in _iter_processes() if _is_ollama_process(proc)]


def _iter_processes() -> Iterable[psutil.Process]:
    return psutil.process_iter(["name"])


def _is_ollama_process(process: psutil.Process) -> bool:
    name = process.info.get("name")
    if not isinstance(name, str):
        return False

    return name.lower().startswith("ollama")
