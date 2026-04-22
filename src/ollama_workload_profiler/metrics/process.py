from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import psutil


def find_ollama_processes() -> list[psutil.Process]:
    processes = list(_iter_processes())
    root_processes = [process for process in processes if _is_ollama_process(process)]
    root_pids = {
        pid
        for process in root_processes
        if (pid := _process_info_int(process, "pid")) is not None
    }
    if not root_pids:
        return root_processes

    included_pids = set(root_pids)
    parent_by_pid = {
        pid: ppid
        for process in processes
        if (pid := _process_info_int(process, "pid")) is not None
        and (ppid := _process_info_int(process, "ppid")) is not None
    }

    changed = True
    while changed:
        changed = False
        for pid, ppid in parent_by_pid.items():
            if pid not in included_pids and ppid in included_pids:
                included_pids.add(pid)
                changed = True

    return [
        process
        for process in processes
        if (pid := _process_info_int(process, "pid")) is not None and pid in included_pids
    ]


def _iter_processes() -> Iterable[psutil.Process]:
    return psutil.process_iter(["name", "pid", "ppid"])


def _is_ollama_process(process: psutil.Process) -> bool:
    name = process.info.get("name")
    if not isinstance(name, str):
        return False

    return name.lower().startswith("ollama")


def _process_info_int(process: Any, key: str) -> int | None:
    info = getattr(process, "info", None)
    if not isinstance(info, dict):
        return None
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
