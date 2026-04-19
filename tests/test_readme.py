from pathlib import Path


README_PATH = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_documents_editable_install_workflow() -> None:
    readme = README_PATH.read_text(encoding="utf-8")

    assert "PYTHONPATH=src" not in readme
    assert "python scripts/bootstrap.py" in readme
    assert ".\\.venv\\Scripts\\python.exe -m ollama_workload_profiler doctor" in readme
    assert "python -m ollama_workload_profiler" in readme
    assert "owp doctor" in readme
    assert "editable install" in readme.lower()


def test_readme_documents_quick_recipes() -> None:
    readme = README_PATH.read_text(encoding="utf-8")

    assert "## Quick recipes" in readme
    assert "owp profile --model llama3.2 --contexts 4096 --benchmark-types smoke,ttft --yes" in readme
    assert "owp compare results/session-a results/session-b" in readme
    assert "owp compare results/session-a results/session-b --strict --all-metrics" in readme
