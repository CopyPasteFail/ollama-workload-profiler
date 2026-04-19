# OWP Compare Design

## Status

Approved for narrow P1 implementation.

## Scope

Add a read-only `owp compare <baseline-session-dir> <candidate-session-dir>` command that compares exactly two completed benchmark session directories from existing artifacts. The command does not execute benchmarks, mutate artifacts, create charts, maintain history, or make statistical significance claims.

## Spec Alignment Notes

- Session-summary metrics render as unavailable with an explicit reason when either side lacks the field.
- Strict-mode failure mapping is fixed by warning severity and category, not inferred implicitly by callers.

