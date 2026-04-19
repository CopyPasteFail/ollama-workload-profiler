from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_EXECUTION_SETTINGS: dict[str, Any] = {
    "seed": 42,
    "temperature": 0.0,
    "repetitions": 1,
}

ALLOWED_EXECUTION_SETTINGS_KEYS = {
    "seed",
    "temperature",
    "top_p",
    "repetitions",
    "warmup_runs",
    "warmup_enabled",
}


def normalize_execution_settings(
    execution_settings: Mapping[str, Any] | None,
) -> dict[str, Any]:
    normalized = dict(DEFAULT_EXECUTION_SETTINGS)
    if execution_settings is not None:
        unknown_keys = set(execution_settings) - ALLOWED_EXECUTION_SETTINGS_KEYS
        if unknown_keys:
            unknown_key = sorted(unknown_keys)[0]
            raise ValueError(f"execution_settings.{unknown_key} is not a supported setting")
        normalized.update(execution_settings)

    seed = normalized.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("execution_settings.seed must be an integer")

    temperature = normalized.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("execution_settings.temperature must be a number between 0.0 and 2.0")
    if not 0.0 <= float(temperature) <= 2.0:
        raise ValueError("execution_settings.temperature must be a number between 0.0 and 2.0")

    top_p = normalized.get("top_p")
    if top_p is not None:
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
            raise ValueError("execution_settings.top_p must be a number between 0.0 and 1.0")
        if not 0.0 <= float(top_p) <= 1.0:
            raise ValueError("execution_settings.top_p must be a number between 0.0 and 1.0")

    warmup_runs = normalized.get("warmup_runs")
    if warmup_runs is not None:
        if isinstance(warmup_runs, bool) or not isinstance(warmup_runs, int) or warmup_runs < 1:
            raise ValueError("execution_settings.warmup_runs must be a positive integer")

    warmup_enabled = normalized.get("warmup_enabled")
    if warmup_enabled is not None and not isinstance(warmup_enabled, bool):
        raise ValueError("execution_settings.warmup_enabled must be a boolean")

    repetitions = normalized.get("repetitions")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("execution_settings.repetitions must be a positive integer")

    return normalized
