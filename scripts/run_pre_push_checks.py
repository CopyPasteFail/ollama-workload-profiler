from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKS: list[tuple[str, list[str]]] = [
    ("Ruff lint", [sys.executable, "-m", "ruff", "check", "."]),
    ("Compile check", [sys.executable, "-m", "compileall", "-q", "src", "tests"]),
    ("Pytest", [sys.executable, "-m", "pytest", "-q"]),
]


def main() -> int:
    for label, command in CHECKS:
        print(f"[pre-push] {label}...")
        completed = subprocess.run(command, cwd=PROJECT_ROOT)
        if completed.returncode != 0:
            print(
                f"[pre-push] {label} failed with exit code {completed.returncode}.",
                file=sys.stderr,
            )
            return completed.returncode
    print("[pre-push] All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
