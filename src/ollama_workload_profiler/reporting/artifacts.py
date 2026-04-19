from __future__ import annotations

import csv
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel

from ..methodology import BENCHMARK_METHODOLOGY_VERSION


ARTIFACT_FILENAMES = (
    "plan.json",
    "expanded_plan.json",
    "environment.json",
    "raw.jsonl",
    "raw.csv",
    "summary.json",
    "report.md",
)


def initialize_session_artifacts(
    base_dir: Path,
    *,
    session_timestamp: datetime | None = None,
    plan: Mapping[str, Any] | BaseModel,
    expanded_plan: Sequence[Mapping[str, Any] | BaseModel],
    environment: Mapping[str, Any] | BaseModel,
) -> Path:
    output_root = _validate_base_dir(base_dir)
    session_dir = _create_session_dir(output_root, session_timestamp=session_timestamp)

    plan_payload = _with_methodology_version(_to_json_payload(plan))
    expanded_plan_payload = [_to_json_payload(item) for item in expanded_plan]
    environment_payload = _with_methodology_version(_to_json_payload(environment))

    _write_json(session_dir / "plan.json", plan_payload)
    _write_json(session_dir / "expanded_plan.json", expanded_plan_payload)
    _write_json(session_dir / "environment.json", environment_payload)
    _write_raw_jsonl(session_dir / "raw.jsonl", [])
    _write_raw_csv(session_dir / "raw.csv", [])

    return session_dir


def append_run_artifact(
    session_dir: Path,
    *,
    run: Mapping[str, Any] | BaseModel,
) -> None:
    row = _to_json_payload(run)
    with (session_dir / "raw.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def finalize_session_artifacts(
    session_dir: Path,
    *,
    runs: Sequence[Mapping[str, Any] | BaseModel],
    summary: Mapping[str, Any] | BaseModel,
    report_markdown: str,
) -> None:
    raw_rows = [_to_json_payload(run) for run in runs]
    summary_payload = _with_artifact_manifest(summary)

    _write_raw_csv(session_dir / "raw.csv", raw_rows)
    _write_json(session_dir / "summary.json", summary_payload)
    (session_dir / "report.md").write_text(report_markdown, encoding="utf-8")


def write_session_artifacts(
    base_dir: Path,
    *,
    session_timestamp: datetime | None = None,
    plan: Mapping[str, Any] | BaseModel,
    expanded_plan: Sequence[Mapping[str, Any] | BaseModel],
    environment: Mapping[str, Any] | BaseModel,
    runs: Sequence[Mapping[str, Any] | BaseModel],
    summary: Mapping[str, Any] | BaseModel,
    report_markdown: str,
) -> Path:
    session_dir = initialize_session_artifacts(
        base_dir,
        session_timestamp=session_timestamp,
        plan=plan,
        expanded_plan=expanded_plan,
        environment=environment,
    )
    for run in runs:
        append_run_artifact(session_dir, run=run)
    finalize_session_artifacts(
        session_dir,
        runs=runs,
        summary=summary,
        report_markdown=report_markdown,
    )

    return session_dir


def _validate_base_dir(base_dir: Path) -> Path:
    resolved = Path(base_dir)
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError("base_dir must be an existing directory")
    return resolved


def _create_session_dir(base_dir: Path, *, session_timestamp: datetime | None) -> Path:
    timestamp = session_timestamp or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    session_name = f"session-{timestamp.strftime('%Y-%m-%d_T%H-%M-%SZ')}"

    candidate = base_dir / session_name
    suffix = 1
    while candidate.exists():
        candidate = base_dir / f"{session_name}-{suffix:02d}"
        suffix += 1

    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def _to_json_payload(value: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return deepcopy(value.model_dump(mode="json"))
    return deepcopy(dict(value))


def _with_artifact_manifest(summary: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
    summary_payload = _with_methodology_version(_to_json_payload(summary))
    summary_payload["artifacts"] = {name: name for name in ARTIFACT_FILENAMES}
    return summary_payload


def _with_methodology_version(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("benchmark_methodology_version", BENCHMARK_METHODOLOGY_VERSION)
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_raw_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_raw_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened_rows = [_flatten_row(row) for row in rows]
    fieldnames: list[str] = []
    for row in flattened_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in flattened_rows:
            writer.writerow(row)


def _flatten_row(payload: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, value in payload.items():
        flattened_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_row(value, prefix=flattened_key))
            continue
        if isinstance(value, list):
            flattened[flattened_key] = json.dumps(value, sort_keys=True)
            continue
        flattened[flattened_key] = "" if value is None else str(value)
    return flattened
