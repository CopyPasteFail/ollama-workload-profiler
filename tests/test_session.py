import json
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
import threading
import time

import pytest
from pydantic import ValidationError

from ollama_workload_profiler.models.plan import BenchmarkSessionPlan, BenchmarkType
from ollama_workload_profiler.models.plan import PlannedRun
from ollama_workload_profiler.models.summary import ModelSummary, ReportSummary
from ollama_workload_profiler.models.verdicts import Verdict, VerdictLabel
from ollama_workload_profiler.models.results import RunResult, RunState
from ollama_workload_profiler.benchmarks import (
    BenchmarkExecutionStopped,
    BenchmarkRunner,
    ExecutionMode,
    ExecutionRequest,
    ExecutionResult,
    build_scenarios_for_benchmark as build_benchmark_scenarios,
    resolve_benchmark_family,
)
from ollama_workload_profiler.metrics.process import find_ollama_processes
from ollama_workload_profiler.metrics.sampler import PollingProcessSampler, SamplePoint
from ollama_workload_profiler.metrics.phases import compute_phase_peaks
from ollama_workload_profiler.prompts.fixtures import (
    MULTI_TURN_CHAT_TURNS,
    STATIC_CODE_SAMPLE,
    STATIC_SUMMARY_TEXT,
)
from ollama_workload_profiler.prompts.scenarios import (
    MultiTurnChatPromptPayload,
    TextPromptPayload,
    build_scenarios_for_benchmark,
)
from ollama_workload_profiler.session import (
    _OllamaDispatcher,
    build_profile_session_plan,
    expand_session_plan,
    run_profile_session,
)


def test_benchmark_session_plan_serializes_to_json() -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[4096, 8192],
        benchmark_types=[BenchmarkType.SMOKE, BenchmarkType.TTFT],
        repetitions=2,
    )

    payload = plan.model_dump(mode="json")

    assert payload["model_name"] == "llama3.2"
    assert payload["contexts"] == [4096, 8192]
    assert payload["benchmark_types"] == ["smoke", "ttft"]


def test_benchmark_session_plan_rejects_blank_model_name() -> None:
    with pytest.raises(ValidationError):
        BenchmarkSessionPlan(
            model_name=" ",
            contexts=[4096],
            benchmark_types=[BenchmarkType.SMOKE],
        )


def test_benchmark_session_plan_rejects_empty_contexts() -> None:
    with pytest.raises(ValidationError):
        BenchmarkSessionPlan(
            model_name="llama3.2",
            contexts=[],
            benchmark_types=[BenchmarkType.SMOKE],
        )


def test_benchmark_session_plan_rejects_empty_benchmark_types() -> None:
    with pytest.raises(ValidationError):
        BenchmarkSessionPlan(
            model_name="llama3.2",
            contexts=[4096],
            benchmark_types=[],
        )


def test_expand_session_plan_uses_fixed_benchmark_execution_order_and_is_deterministic() -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[8192, 4096],
        benchmark_types=[BenchmarkType.CONTEXT_SCALING, BenchmarkType.SMOKE],
        repetitions=1,
    )

    first_pass = expand_session_plan(plan)
    second_pass = expand_session_plan(plan)

    assert [run.run_index for run in first_pass] == list(range(1, len(first_pass) + 1))
    assert [run.run_id for run in first_pass] == [run.run_id for run in second_pass]
    assert [
        (
            run.model_name,
            run.context_size,
            run.benchmark_type,
            run.scenario_id,
            run.repetition_index,
        )
        for run in first_pass
    ] == [
        (
            run.model_name,
            run.context_size,
            run.benchmark_type,
            run.scenario_id,
            run.repetition_index,
        )
        for run in second_pass
    ]
    assert len({run.run_id for run in first_pass}) == len(first_pass)
    assert first_pass[0].run_id != "run-0001"
    assert [benchmark_type for benchmark_type, _group in groupby(run.benchmark_type for run in first_pass)] == [
        BenchmarkType.SMOKE,
        BenchmarkType.CONTEXT_SCALING,
        BenchmarkType.SMOKE,
        BenchmarkType.CONTEXT_SCALING,
    ]


def test_expand_session_plan_assigns_repetition_ordering_metadata() -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE],
        repetitions=2,
    )

    runs = expand_session_plan(plan)

    assert [run.run_index for run in runs] == [1, 2]
    assert [run.repetition_index for run in runs] == [1, 2]
    assert {run.scenario_id for run in runs} == {"smoke-basic-v1"}
    assert runs[0].run_id != runs[1].run_id


def test_expand_session_plan_keeps_run_ids_unique_for_repeated_contexts() -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[4096, 4096],
        benchmark_types=[BenchmarkType.SMOKE],
        repetitions=1,
    )

    runs = expand_session_plan(plan)

    assert [run.context_index for run in runs] == [1, 2]
    assert runs[0].run_id != runs[1].run_id
    assert len({run.run_id for run in runs}) == len(runs)


def test_expand_session_plan_keeps_run_ids_unique_when_benchmark_types_repeat_in_the_plan() -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE, BenchmarkType.SMOKE],
        repetitions=1,
    )

    runs = expand_session_plan(plan)

    assert [run.benchmark_type_index for run in runs] == [1]
    assert [benchmark_type for benchmark_type, _group in groupby(run.benchmark_type for run in runs)] == [
        BenchmarkType.SMOKE,
    ]
    assert len({run.run_id for run in runs}) == len(runs)


def test_expand_session_plan_normalizes_duplicate_benchmark_types_into_fixed_order() -> None:
    plan = BenchmarkSessionPlan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[
            BenchmarkType.CONTEXT_SCALING,
            BenchmarkType.SMOKE,
            BenchmarkType.CONTEXT_SCALING,
        ],
        repetitions=1,
    )

    runs = expand_session_plan(plan)

    assert [benchmark_type for benchmark_type, _group in groupby(run.benchmark_type for run in runs)] == [
        BenchmarkType.SMOKE,
        BenchmarkType.CONTEXT_SCALING,
    ]


def test_run_profile_session_uses_supplied_expanded_plan_order_and_appends_raw_rows_incrementally(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = build_profile_session_plan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE, BenchmarkType.PROMPT_SCALING],
        repetitions=1,
    )
    expanded_plan = list(reversed(expand_session_plan(plan)))
    observed_run_ids: list[str] = []

    class FakeRunner:
        def __init__(self, **_: object) -> None:
            return None

        def run(self, planned_run: PlannedRun) -> RunResult:
            session_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
            assert len(session_dirs) == 1
            session_dir = session_dirs[0]

            assert (session_dir / "plan.json").exists()
            assert (session_dir / "expanded_plan.json").exists()
            assert (session_dir / "environment.json").exists()
            assert (session_dir / "raw.jsonl").exists()
            assert (session_dir / "raw.csv").exists()

            expanded_payload = json.loads((session_dir / "expanded_plan.json").read_text(encoding="utf-8"))
            assert [item["run_id"] for item in expanded_payload] == [item.run_id for item in expanded_plan]

            raw_lines = (session_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
            assert len(raw_lines) == len(observed_run_ids)

            observed_run_ids.append(planned_run.run_id)
            return RunResult(
                run_id=planned_run.run_id,
                run_index=planned_run.run_index,
                model_name=planned_run.model_name,
                context_size=planned_run.context_size,
                context_index=planned_run.context_index,
                benchmark_type=planned_run.benchmark_type,
                benchmark_type_index=planned_run.benchmark_type_index,
                scenario_id=planned_run.scenario_id,
                scenario_index=planned_run.scenario_index,
                scenario_version=planned_run.scenario_version,
                repetition_index=planned_run.repetition_index,
                scenario_name=planned_run.scenario_name,
                state=RunState.COMPLETED,
                elapsed_ms=10.0,
                metrics={"tokens_per_second": 5.0},
            )

    class FakeClient:
        def list_models(self) -> list[str]:
            return [plan.model_name]

    monkeypatch.setattr("ollama_workload_profiler.session.BenchmarkRunner", FakeRunner)

    result = run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        expanded_plan=expanded_plan,
        session_timestamp=datetime(2026, 4, 19, 10, 0, 0, tzinfo=timezone.utc),
    )

    assert observed_run_ids == [run.run_id for run in expanded_plan]

    raw_rows = (result.session_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw_rows) == len(expanded_plan)
    assert [json.loads(row)["run_id"] for row in raw_rows] == observed_run_ids
    assert {json.loads(row)["scenario_version"] for row in raw_rows} == {"v1"}

    environment_payload = json.loads((result.session_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment_payload["session_started_at"] == "2026-04-19T10:00:00+00:00"


def test_run_profile_session_persists_phase_peaks_from_real_sampler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = build_profile_session_plan(
        model_name="llama3.2",
        contexts=[4096],
        benchmark_types=[BenchmarkType.SMOKE],
        repetitions=1,
    )

    class FakeMemoryInfo:
        def __init__(self, rss: int) -> None:
            self.rss = rss

    class FakeProcess:
        def __init__(self) -> None:
            self._cpu_values = iter([5.0, 35.0, 12.0])

        def memory_info(self) -> FakeMemoryInfo:
            return FakeMemoryInfo(192 * 1024 * 1024)

        def cpu_percent(self, interval: float | None = None) -> float:
            return next(self._cpu_values, 12.0)

    class FakeClient:
        def list_models(self) -> list[str]:
            return [plan.model_name]

        def generate(self, *, model: str, prompt: str, options: dict[str, object]) -> dict[str, object]:
            time.sleep(0.01)
            return {
                "response": "ok",
                "total_duration": 1_000_000,
                "eval_count": 4,
                "eval_duration": 250_000,
                "prompt_eval_count": 8,
                "prompt_eval_duration": 500_000,
            }

    fake_processes = [FakeProcess()]
    monkeypatch.setattr(
        "ollama_workload_profiler.session.find_ollama_processes",
        lambda: fake_processes,
    )

    result = run_profile_session(
        plan=plan,
        client=FakeClient(),
        output_dir=tmp_path,
        session_timestamp=datetime(2026, 4, 19, 11, 0, 0, tzinfo=timezone.utc),
    )

    raw_rows = [
        json.loads(line)
        for line in (result.session_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(raw_rows) == 1
    assert raw_rows[0]["metrics"]["phase_peaks"]["generation"]["rss_mb"] == 192.0
    assert raw_rows[0]["metrics"]["phase_peaks"]["generation"]["cpu_percent"] == 35.0


def test_compute_phase_peaks_groups_samples_by_phase() -> None:
    peaks = compute_phase_peaks(
        samples=[
            {"phase": "load", "rss_mb": 100, "cpu_percent": 5},
            {"phase": "generation", "rss_mb": 150, "cpu_percent": 35},
        ]
    )

    assert peaks["generation"].rss_mb == 150
    assert peaks["generation"].cpu_percent == 35
    assert peaks["load"].rss_mb == 100


def test_compute_phase_peaks_breaks_rss_ties_by_higher_cpu_percent() -> None:
    peaks = compute_phase_peaks(
        samples=[
            {"phase": "generation", "rss_mb": 150, "cpu_percent": 12},
            {"phase": "generation", "rss_mb": 150, "cpu_percent": 35},
        ]
    )

    assert peaks["generation"].rss_mb == 150
    assert peaks["generation"].cpu_percent == 35


@pytest.mark.parametrize(
    "sample",
    [
        {"phase": None, "rss_mb": 100, "cpu_percent": 5},
        {"phase": True, "rss_mb": 100, "cpu_percent": 5},
        {"phase": "load", "rss_mb": float("nan"), "cpu_percent": 5},
        {"phase": "load", "rss_mb": 100, "cpu_percent": float("inf")},
        {"phase": "load", "rss_mb": True, "cpu_percent": 5},
    ],
)
def test_compute_phase_peaks_rejects_malformed_telemetry(sample: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        compute_phase_peaks([sample])


def test_find_ollama_processes_filters_process_names_case_insensitively(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        def __init__(self, name: str) -> None:
            self.info = {"name": name}

    monkeypatch.setattr(
        "ollama_workload_profiler.metrics.process.psutil.process_iter",
        lambda attrs: [
            FakeProcess("ollama"),
            FakeProcess("Ollama-helper"),
            FakeProcess("python"),
        ],
    )

    processes = find_ollama_processes()

    assert [process.info["name"] for process in processes] == ["ollama", "Ollama-helper"]


def test_polling_process_sampler_collects_load_generation_and_post_run_samples() -> None:
    class FakeMemoryInfo:
        def __init__(self, rss: int) -> None:
            self.rss = rss

    class FakeProcess:
        def memory_info(self) -> FakeMemoryInfo:
            return FakeMemoryInfo(256 * 1024 * 1024)

        def cpu_percent(self, interval: float | None = None) -> float:
            return 12.5

    planned_run = PlannedRun(
        run_id="run-sampler",
        run_index=1,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.SMOKE,
        benchmark_type_index=1,
        scenario_id="smoke-basic-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="Smoke sanity check",
        scenario_version="v1",
    )
    fake_process = FakeProcess()
    sampler = PollingProcessSampler(
        process_finder=lambda: [fake_process],
        interval_seconds=0.001,
        shutdown_timeout_seconds=0.5,
    )

    sampler.start(planned_run)
    time.sleep(0.01)
    samples = sampler.stop()

    assert samples[0].phase == "load"
    assert samples[-1].phase == "resident_post_run"
    assert any(sample.phase == "generation" for sample in samples)
    assert all(sample.rss_mb == 256.0 for sample in samples)
    assert all(sample.cpu_percent == 12.5 for sample in samples)


def test_polling_process_sampler_waits_for_in_flight_poll_before_stop_returns() -> None:
    entered_generation = threading.Event()
    release_generation = threading.Event()

    class FakeMemoryInfo:
        def __init__(self, rss: int) -> None:
            self.rss = rss

    class FakeProcess:
        def __init__(self) -> None:
            self._cpu_calls = 0

        def memory_info(self) -> FakeMemoryInfo:
            return FakeMemoryInfo(128 * 1024 * 1024)

        def cpu_percent(self, interval: float | None = None) -> float:
            self._cpu_calls += 1
            if self._cpu_calls == 2:
                entered_generation.set()
                release_generation.wait(timeout=1.0)
            return 10.0

    planned_run = PlannedRun(
        run_id="run-sampler-stop",
        run_index=1,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.SMOKE,
        benchmark_type_index=1,
        scenario_id="smoke-basic-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="Smoke sanity check",
        scenario_version="v1",
    )
    fake_process = FakeProcess()
    sampler = PollingProcessSampler(
        process_finder=lambda: [fake_process],
        interval_seconds=0.001,
        shutdown_timeout_seconds=0.5,
    )
    stop_completed = threading.Event()

    sampler.start(planned_run)
    assert entered_generation.wait(timeout=1.0) is True

    stop_thread = threading.Thread(
        target=lambda: (sampler.stop(), stop_completed.set()),
        daemon=True,
    )
    stop_thread.start()
    time.sleep(0.15)

    assert stop_completed.is_set() is False

    release_generation.set()
    stop_thread.join(timeout=1.0)

    assert stop_completed.is_set() is True


def test_polling_process_sampler_stop_returns_when_post_run_snapshot_stalls() -> None:
    allow_post_run = threading.Event()

    class FakeMemoryInfo:
        def __init__(self, rss: int) -> None:
            self.rss = rss

    class FakeProcess:
        def __init__(self) -> None:
            self._cpu_calls = 0

        def memory_info(self) -> FakeMemoryInfo:
            return FakeMemoryInfo(128 * 1024 * 1024)

        def cpu_percent(self, interval: float | None = None) -> float:
            self._cpu_calls += 1
            if self._cpu_calls >= 3:
                allow_post_run.wait(timeout=1.0)
            return 10.0

    planned_run = PlannedRun(
        run_id="run-sampler-post-run",
        run_index=1,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.SMOKE,
        benchmark_type_index=1,
        scenario_id="smoke-basic-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="Smoke sanity check",
        scenario_version="v1",
    )
    fake_process = FakeProcess()
    sampler = PollingProcessSampler(
        process_finder=lambda: [fake_process],
        interval_seconds=0.001,
        shutdown_timeout_seconds=0.05,
    )
    stop_completed = threading.Event()

    sampler.start(planned_run)
    time.sleep(0.01)

    stop_thread = threading.Thread(
        target=lambda: (sampler.stop(), stop_completed.set()),
        daemon=True,
    )
    stop_thread.start()
    time.sleep(0.2)

    assert stop_completed.is_set() is True


def test_report_summary_uses_typed_model_summaries() -> None:
    summary = ReportSummary(
        model_summaries={
            "llama3.2": ModelSummary(
                summary="solid",
                metrics={"ttft_ms": 123.4},
            )
        }
    )

    payload = summary.model_dump(mode="json")

    assert payload["model_summaries"]["llama3.2"]["summary"] == "solid"
    assert payload["model_summaries"]["llama3.2"]["metrics"]["ttft_ms"] == 123.4


def test_verdict_label_is_machine_friendly_but_still_displayable() -> None:
    assert VerdictLabel.GOOD_FIT.value == "good_fit"
    assert VerdictLabel.GOOD_FIT.display_label == "Good fit"


def test_verdict_serializes_machine_friendly_label() -> None:
    verdict = Verdict(
        label=VerdictLabel.GOOD_FIT,
        rationale="Core metrics are within acceptable bands.",
        supporting_metrics={"ttft_ms": 123.4},
    )

    payload = verdict.model_dump(mode="json")

    assert payload["label"] == "good_fit"


def test_run_result_has_stable_ordering_fields() -> None:
    result = RunResult(
        run_id="run-0001",
        run_index=1,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type="smoke",
        benchmark_type_index=1,
        scenario_id="smoke-basic-v1",
        scenario_index=1,
        scenario_version="v1",
        state=RunState.PLANNED,
        repetition_index=1,
    )

    assert result.run_index == 1
    assert result.run_id == "run-0001"
    assert result.context_index == 1
    assert result.benchmark_type_index == 1
    assert result.scenario_index == 1
    assert result.scenario_version == "v1"


def test_prompt_scaling_builds_three_scenarios() -> None:
    scenarios = build_scenarios_for_benchmark(BenchmarkType.PROMPT_SCALING, 4096)
    scenarios_by_id = {item.scenario_id: item for item in scenarios}
    small_prompt = scenarios_by_id["prompt-scaling-small-v1"].prompt_payload
    medium_prompt = scenarios_by_id["prompt-scaling-medium-v1"].prompt_payload
    large_prompt = scenarios_by_id["prompt-scaling-large-v1"].prompt_payload

    assert set(scenarios_by_id) == {
        "prompt-scaling-small-v1",
        "prompt-scaling-medium-v1",
        "prompt-scaling-large-v1",
    }
    assert isinstance(small_prompt, TextPromptPayload)
    assert len(small_prompt.text) == 1024
    assert len(medium_prompt.text) == 4096
    assert len(large_prompt.text) == 16384


def test_use_case_profiles_use_deterministic_fixtures() -> None:
    scenarios = build_scenarios_for_benchmark(BenchmarkType.USE_CASE_PROFILES, 4096)
    scenarios_by_id = {item.scenario_id: item for item in scenarios}
    summary_payload = scenarios_by_id["use-case-profiles-long-document-summary-v1"].prompt_payload
    code_payload = scenarios_by_id["use-case-profiles-code-explanation-v1"].prompt_payload
    multi_turn_payload = scenarios_by_id["use-case-profiles-multi-turn-chat-v1"].prompt_payload

    assert set(scenarios_by_id) == {
        "use-case-profiles-quick-qa-v1",
        "use-case-profiles-long-document-summary-v1",
        "use-case-profiles-code-explanation-v1",
        "use-case-profiles-structured-extraction-v1",
        "use-case-profiles-long-form-generation-v1",
        "use-case-profiles-multi-turn-chat-v1",
    }
    assert summary_payload == TextPromptPayload(STATIC_SUMMARY_TEXT)
    assert code_payload == TextPromptPayload(STATIC_CODE_SAMPLE)
    assert isinstance(multi_turn_payload, MultiTurnChatPromptPayload)
    assert multi_turn_payload.turns == tuple(MULTI_TURN_CHAT_TURNS)
    assert multi_turn_payload.turns == (
        "Summarize the release note in one sentence.",
        "Rewrite it for a technical audience.",
        "Extract three action items.",
    )


def test_benchmark_family_resolution_keeps_ttft_as_distinct_execution_mode() -> None:
    smoke_family = resolve_benchmark_family(BenchmarkType.SMOKE)
    ttft_family = resolve_benchmark_family(BenchmarkType.TTFT)

    smoke_scenarios = build_benchmark_scenarios(BenchmarkType.SMOKE, 4096)
    ttft_scenarios = build_benchmark_scenarios(BenchmarkType.TTFT, 4096)

    assert smoke_family.execution_mode is ExecutionMode.GENERATE
    assert ttft_family.execution_mode is ExecutionMode.TTFT
    assert smoke_scenarios[0].scenario_id == "smoke-basic-v1"
    assert ttft_scenarios[0].scenario_id == "ttft-basic-v1"


def test_ttft_family_builds_two_deterministic_scenarios() -> None:
    scenarios = build_benchmark_scenarios(BenchmarkType.TTFT, 4096)

    assert [scenario.scenario_id for scenario in scenarios] == [
        "ttft-basic-v1",
        "ttft-chat-v1",
    ]


def test_stress_family_builds_two_deterministic_scenarios() -> None:
    scenarios = build_benchmark_scenarios(BenchmarkType.STRESS, 4096)

    assert [scenario.scenario_id for scenario in scenarios] == [
        "stress-sustained-load-v1",
        "stress-burst-load-v1",
    ]


def test_output_scaling_long_scenario_uses_release_target_band() -> None:
    scenarios = build_benchmark_scenarios(BenchmarkType.OUTPUT_SCALING, 4096)
    long_scenario = next(scenario for scenario in scenarios if scenario.scenario_id == "output-scaling-long-v1")

    assert long_scenario.target_output_tokens == 768


def test_cold_warm_family_declares_requested_prep_behavior() -> None:
    scenarios = build_benchmark_scenarios(BenchmarkType.COLD_WARM, 4096)

    assert [(scenario.scenario_id, scenario.profile_tag) for scenario in scenarios] == [
        ("cold-warm-cold-start-v1", "cold_start"),
        ("cold-warm-warm-start-v1", "warm_start"),
    ]


def test_ollama_dispatcher_uses_stream_generate_for_ttft_and_measures_first_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned_run = PlannedRun(
        run_id="run-ttft-dispatch",
        run_index=1,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.TTFT,
        benchmark_type_index=1,
        scenario_id="ttft-basic-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="TTFT probe",
        scenario_version="v1",
    )
    scenario = build_benchmark_scenarios(BenchmarkType.TTFT, 4096)[0]

    class FakeClient:
        def stream_generate(self, *, model: str, prompt: str, options: dict[str, object]) -> list[dict[str, object]]:
            assert model == planned_run.model_name
            assert options["num_ctx"] == planned_run.context_size
            assert options["num_predict"] == 8
            return [
                {"response": "o", "done": False},
                {"response": "ok", "done": True, "eval_count": 4, "eval_duration": 250_000},
            ]

    perf_values = iter([10.0, 10.05, 10.2])
    monkeypatch.setattr("ollama_workload_profiler.session.perf_counter", lambda: next(perf_values))

    result = _OllamaDispatcher(FakeClient()).execute(
        ExecutionRequest(
            run=planned_run,
            scenario=scenario,
            execution_mode=ExecutionMode.TTFT,
        )
    )

    assert result.metrics["ttft_ms"] == 50.0
    assert result.metrics["ttft_first_token_received"] is True
    assert result.metrics["tokens_per_second"] == 16000.0


def test_ollama_dispatcher_records_missing_first_token_for_empty_ttft_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned_run = PlannedRun(
        run_id="run-ttft-empty",
        run_index=2,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.TTFT,
        benchmark_type_index=1,
        scenario_id="ttft-basic-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="TTFT probe",
        scenario_version="v1",
    )
    scenario = build_benchmark_scenarios(BenchmarkType.TTFT, 4096)[0]

    class FakeClient:
        def stream_generate(self, *, model: str, prompt: str, options: dict[str, object]) -> list[dict[str, object]]:
            return []

    perf_values = iter([20.0, 20.1])
    monkeypatch.setattr("ollama_workload_profiler.session.perf_counter", lambda: next(perf_values))

    result = _OllamaDispatcher(FakeClient()).execute(
        ExecutionRequest(
            run=planned_run,
            scenario=scenario,
            execution_mode=ExecutionMode.TTFT,
        )
    )

    assert "ttft_ms" not in result.metrics
    assert result.metrics["ttft_first_token_received"] is False
    assert result.elapsed_ms == 100.0


def test_ollama_dispatcher_uses_stream_chat_for_chat_ttft_scenarios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned_run = PlannedRun(
        run_id="run-ttft-chat",
        run_index=3,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.TTFT,
        benchmark_type_index=1,
        scenario_id="ttft-chat-v1",
        scenario_index=2,
        repetition_index=1,
        scenario_name="TTFT chat probe",
        scenario_version="v1",
    )
    scenario = build_benchmark_scenarios(BenchmarkType.TTFT, 4096)[1]
    stream_chat_calls: list[list[dict[str, object]]] = []

    class FakeClient:
        def stream_chat(self, *, model: str, messages: list[dict[str, object]], options: dict[str, object]) -> list[dict[str, object]]:
            assert model == planned_run.model_name
            assert options["num_ctx"] == planned_run.context_size
            assert options["num_predict"] == 8
            stream_chat_calls.append(messages)
            return [
                {"message": {"content": "", "thinking": "Thinking"}, "done": False},
                {"message": {"content": "", "thinking": ""}, "done": True},
            ]

    perf_values = iter([30.0, 30.02, 30.1])
    monkeypatch.setattr("ollama_workload_profiler.session.perf_counter", lambda: next(perf_values))

    result = _OllamaDispatcher(FakeClient()).execute(
        ExecutionRequest(
            run=planned_run,
            scenario=scenario,
            execution_mode=ExecutionMode.TTFT,
        )
    )

    assert len(stream_chat_calls) == 1
    assert [message["content"] for message in stream_chat_calls[0]] == [
        "Reply with a single token.",
        "Reply again with the same single token.",
    ]
    assert result.metrics["ttft_ms"] == 20.0
    assert result.metrics["ttft_first_token_received"] is True


def test_ollama_dispatcher_does_not_count_empty_non_content_ttft_chunks_as_first_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned_run = PlannedRun(
        run_id="run-ttft-empty-chat",
        run_index=4,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.TTFT,
        benchmark_type_index=1,
        scenario_id="ttft-chat-v1",
        scenario_index=2,
        repetition_index=1,
        scenario_name="TTFT chat probe",
        scenario_version="v1",
    )
    scenario = build_benchmark_scenarios(BenchmarkType.TTFT, 4096)[1]

    class FakeClient:
        def stream_chat(self, *, model: str, messages: list[dict[str, object]], options: dict[str, object]) -> list[dict[str, object]]:
            return [
                {"message": {"content": "", "thinking": ""}, "done": False},
                {"message": {"content": "", "thinking": ""}, "done": True},
            ]

    perf_values = iter([40.0, 40.1])
    monkeypatch.setattr("ollama_workload_profiler.session.perf_counter", lambda: next(perf_values))

    result = _OllamaDispatcher(FakeClient()).execute(
        ExecutionRequest(
            run=planned_run,
            scenario=scenario,
            execution_mode=ExecutionMode.TTFT,
        )
    )

    assert "ttft_ms" not in result.metrics
    assert result.metrics["ttft_first_token_received"] is False


def test_benchmark_runner_finalizes_completed_run_from_execution_facts() -> None:
    planned_run = PlannedRun(
        run_id="run-1234",
        run_index=1,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.SMOKE,
        benchmark_type_index=1,
        scenario_id="smoke-basic-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="Smoke sanity check",
    )
    state_transitions: list[RunState] = []
    dispatched_modes: list[ExecutionMode] = []

    class FakeSampler:
        def __init__(self) -> None:
            self.started = False
            self.stopped = False

        def start(self, run: PlannedRun) -> None:
            self.started = True
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[SamplePoint]:
            self.stopped = True
            return [SamplePoint(phase="generation", rss_mb=128.0, cpu_percent=44.0)]

    class FakeDispatcher:
        def execute(self, request: object) -> ExecutionResult:
            dispatched_modes.append(request.execution_mode)
            return ExecutionResult(
                elapsed_ms=42.5,
                metrics={"tokens_per_second": 12.0},
            )

    sampler = FakeSampler()
    runner = BenchmarkRunner(
        dispatcher=FakeDispatcher(),
        sampler_factory=lambda: sampler,
        on_state_change=lambda _run, state: state_transitions.append(state),
    )

    result = runner.run(planned_run)

    assert sampler.started is True
    assert sampler.stopped is True
    assert dispatched_modes == [ExecutionMode.GENERATE]
    assert state_transitions == [RunState.STARTING, RunState.COMPLETED]
    assert result.state is RunState.COMPLETED
    assert result.elapsed_ms == 42.5
    assert result.metrics["tokens_per_second"] == 12.0
    assert result.metrics["phase_peaks"]["generation"]["rss_mb"] == 128.0


def test_benchmark_runner_uses_ttft_execution_mode_and_classifies_stopped_runs() -> None:
    planned_run = PlannedRun(
        run_id="run-ttft",
        run_index=2,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.TTFT,
        benchmark_type_index=1,
        scenario_id="ttft-basic-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="TTFT probe",
    )
    state_transitions: list[RunState] = []
    dispatched_modes: list[ExecutionMode] = []

    class FakeSampler:
        def start(self, run: PlannedRun) -> None:
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[SamplePoint]:
            return []

    class FakeDispatcher:
        def execute(self, request: object) -> ExecutionResult:
            dispatched_modes.append(request.execution_mode)
            raise BenchmarkExecutionStopped("cancelled")

    runner = BenchmarkRunner(
        dispatcher=FakeDispatcher(),
        sampler_factory=FakeSampler,
        on_state_change=lambda _run, state: state_transitions.append(state),
    )

    result = runner.run(planned_run)

    assert dispatched_modes == [ExecutionMode.TTFT]
    assert state_transitions == [RunState.STARTING, RunState.STOPPED]
    assert result.state is RunState.STOPPED
    assert result.metrics["stop_reason"] == "cancelled"


def test_benchmark_runner_returns_failed_result_when_sampler_stop_raises() -> None:
    planned_run = PlannedRun(
        run_id="run-stop-error",
        run_index=3,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.SMOKE,
        benchmark_type_index=1,
        scenario_id="smoke-basic-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="Smoke sanity check",
    )
    state_transitions: list[RunState] = []

    class FakeSampler:
        def start(self, run: PlannedRun) -> None:
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[SamplePoint]:
            raise RuntimeError("sampler shutdown failed")

    class FakeDispatcher:
        def execute(self, request: object) -> ExecutionResult:
            assert request.execution_mode is ExecutionMode.GENERATE
            return ExecutionResult(
                elapsed_ms=19.5,
                metrics={"tokens_per_second": 8.0},
            )

    runner = BenchmarkRunner(
        dispatcher=FakeDispatcher(),
        sampler_factory=FakeSampler,
        on_state_change=lambda _run, state: state_transitions.append(state),
    )

    result = runner.run(planned_run)

    assert state_transitions == [RunState.STARTING, RunState.FAILED]
    assert result.state is RunState.FAILED
    assert result.elapsed_ms == 19.5
    assert result.metrics["tokens_per_second"] == 8.0
    assert result.metrics["execution_outcome_state"] == RunState.COMPLETED.value
    assert result.metrics["finalization_error"] == "sampler shutdown failed"
    assert result.failure is not None
    assert result.failure.kind == "failed"
    assert result.failure.message == "sampler shutdown failed"


def test_benchmark_runner_returns_failed_result_when_telemetry_is_malformed() -> None:
    planned_run = PlannedRun(
        run_id="run-bad-telemetry",
        run_index=4,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.TTFT,
        benchmark_type_index=1,
        scenario_id="ttft-basic-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="TTFT probe",
    )
    state_transitions: list[RunState] = []

    class FakeSampler:
        def start(self, run: PlannedRun) -> None:
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[dict[str, object]]:
            return [{"phase": None, "rss_mb": 100.0, "cpu_percent": 5.0}]

    class FakeDispatcher:
        def execute(self, request: object) -> ExecutionResult:
            assert request.execution_mode is ExecutionMode.TTFT
            raise BenchmarkExecutionStopped("cancelled")

    runner = BenchmarkRunner(
        dispatcher=FakeDispatcher(),
        sampler_factory=FakeSampler,
        on_state_change=lambda _run, state: state_transitions.append(state),
    )

    result = runner.run(planned_run)

    assert state_transitions == [RunState.STARTING, RunState.FAILED]
    assert result.state is RunState.FAILED
    assert result.metrics["stop_reason"] == "cancelled"
    assert result.metrics["execution_outcome_state"] == RunState.STOPPED.value
    assert result.metrics["finalization_error"] == "sample.phase must be a non-empty string"
    assert result.failure is not None
    assert result.failure.kind == "failed"
    assert result.failure.message == "sample.phase must be a non-empty string"


def test_benchmark_runner_persists_cold_warm_prep_metadata() -> None:
    planned_run = PlannedRun(
        run_id="run-cold-warm",
        run_index=5,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.COLD_WARM,
        benchmark_type_index=1,
        scenario_id="cold-warm-cold-start-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="Cold start probe",
        scenario_version="v1",
    )

    class FakeSampler:
        def start(self, run: PlannedRun) -> None:
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[SamplePoint]:
            return []

    class FakeDispatcher:
        def execute(self, request: object) -> ExecutionResult:
            return ExecutionResult(elapsed_ms=33.0, metrics={})

    runner = BenchmarkRunner(
        dispatcher=FakeDispatcher(),
        sampler_factory=FakeSampler,
    )

    result = runner.run(planned_run)

    assert result.state is RunState.COMPLETED
    assert result.metrics["requested_prep_behavior"] == "cold_start"
    assert result.metrics["actual_prep_method"] == "scenario_declared"


def test_benchmark_runner_preserves_cold_warm_prep_metadata_for_stopped_runs() -> None:
    planned_run = PlannedRun(
        run_id="run-cold-warm-stopped",
        run_index=6,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.COLD_WARM,
        benchmark_type_index=1,
        scenario_id="cold-warm-cold-start-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="Cold start probe",
        scenario_version="v1",
    )

    class FakeSampler:
        def start(self, run: PlannedRun) -> None:
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[SamplePoint]:
            return []

    class FakeDispatcher:
        def execute(self, request: object) -> ExecutionResult:
            raise BenchmarkExecutionStopped("cancelled")

    runner = BenchmarkRunner(
        dispatcher=FakeDispatcher(),
        sampler_factory=FakeSampler,
    )

    result = runner.run(planned_run)

    assert result.state is RunState.STOPPED
    assert result.metrics["requested_prep_behavior"] == "cold_start"
    assert result.metrics["actual_prep_method"] == "scenario_declared"


def test_benchmark_runner_preserves_cold_warm_prep_metadata_when_sampler_stop_fails() -> None:
    planned_run = PlannedRun(
        run_id="run-cold-warm-finalization-failed",
        run_index=7,
        model_name="llama3.2",
        context_size=4096,
        context_index=1,
        benchmark_type=BenchmarkType.COLD_WARM,
        benchmark_type_index=1,
        scenario_id="cold-warm-cold-start-v1",
        scenario_index=1,
        repetition_index=1,
        scenario_name="Cold start probe",
        scenario_version="v1",
    )

    class FakeSampler:
        def start(self, run: PlannedRun) -> None:
            assert run.run_id == planned_run.run_id

        def stop(self) -> list[SamplePoint]:
            raise RuntimeError("sampler stop failed")

    class FakeDispatcher:
        def execute(self, request: object) -> ExecutionResult:
            return ExecutionResult(elapsed_ms=12.0, metrics={})

    runner = BenchmarkRunner(
        dispatcher=FakeDispatcher(),
        sampler_factory=FakeSampler,
    )

    result = runner.run(planned_run)

    assert result.state is RunState.FAILED
    assert result.failure is not None
    assert result.failure.phase == "finalization"
    assert result.metrics["requested_prep_behavior"] == "cold_start"
    assert result.metrics["actual_prep_method"] == "scenario_declared"
    assert result.metrics["finalization_error"] == "sampler stop failed"
