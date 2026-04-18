"""Deterministic prompt fixtures used by benchmark scenarios."""

from __future__ import annotations

STATIC_SUMMARY_TEXT = (
    "Release note summary fixture.\n"
    "\n"
    "Section 1: The product adds deterministic benchmark planning and reporting.\n"
    "Section 2: The update keeps fixture text source-controlled and easy to inspect.\n"
    "Section 3: Teams should confirm behavioral changes with focused tests.\n"
)

STATIC_CODE_SAMPLE = (
    "def fib(n: int) -> int:\n"
    "    if n < 2:\n"
    "        return n\n"
    "    return fib(n - 1) + fib(n - 2)\n"
)

MULTI_TURN_CHAT_TURNS: tuple[str, ...] = (
    "Summarize the release note in one sentence.",
    "Rewrite it for a technical audience.",
    "Extract three action items.",
)

PROMPT_SCALING_BASE_TEXT = (
    "Benchmark planning fixture text.\n"
    "This prompt exists only to create deterministic prompt-size variants.\n"
)
