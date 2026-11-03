"""--repeats: ask every case N times, score the mean, expose the spread."""

import threading
from collections import Counter

import pytest
from typer.testing import CliRunner

from evalx import dashboard, store
from evalx.cli import app
from evalx.runner import _aggregate, run_suite
from evalx.schemas import CaseResult

DEMO = "datasets/demo/v1"
TARGET = "examples.demo_app:answer"
FLAKY = "tests.test_repeats:flaky"
ALWAYS_CRASH_CEO = "tests.test_repeats:crash_ceo_once"

_calls: Counter = Counter()
_lock = threading.Lock()


def flaky(question: str) -> str:
    """The demo app on odd attempts, a non-answer on even ones (per question)."""
    from examples.demo_app import answer

    with _lock:
        _calls[question] += 1
        n = _calls[question]
    return answer(question) if n % 2 == 1 else "I'm not sure."


def crash_ceo_once(question: str) -> str:
    """Raises on the first CEO attempt only."""
    from examples.demo_app import answer

    with _lock:
        _calls[question] += 1
        n = _calls[question]
    if "CEO" in question and n == 1:
        raise RuntimeError("flaky upstream")
    return answer(question)


@pytest.fixture(autouse=True)
def _reset_calls():
    _calls.clear()


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALX_STORE", str(tmp_path / "runs.duckdb"))


def _flat(text: str) -> str:
    return " ".join(text.split())


# --- runner -------------------------------------------------------------------


def test_single_attempt_is_unchanged():
    result = run_suite(DEMO, TARGET)
    assert result.repeats == 1
    assert result.mean_score == 0.5
    assert all(r.attempt_scores == [r.score] for r in result.results)
    assert all(r.spread is None for r in result.results)


def test_deterministic_target_has_zero_spread():
    result = run_suite(DEMO, TARGET, repeats=3)
    assert result.repeats == 3
    assert result.mean_score == 0.5  # same as one attempt
    assert all(len(r.attempt_scores) == 3 for r in result.results)
    assert all(r.spread == 0.0 for r in result.results)


def test_flaky_target_scores_mean_and_shows_spread():
    result = run_suite(DEMO, FLAKY, repeats=2)
    q001 = {r.case_id: r for r in result.results}["q001"]
    # Attempt 1: demo answer (1.0); attempt 2: non-answer (0.0).
    assert q001.attempt_scores == [1.0, 0.0]
    assert q001.score == 0.5
    assert q001.spread == 1.0


def test_detail_comes_from_worst_attempt():
    result = run_suite(DEMO, FLAKY, repeats=2)
    q001 = {r.case_id: r for r in result.results}["q001"]
    assert q001.answer == "I'm not sure."  # the 0.0 attempt, the one to debug
    assert q001.check_scores == [0.0]


def test_attempts_run_in_parallel_with_same_result():
    seq = run_suite(DEMO, TARGET, repeats=3)
    par = run_suite(DEMO, TARGET, repeats=3, workers=6)
    assert [r.case_id for r in par.results] == [r.case_id for r in seq.results]
    assert [r.attempt_scores for r in par.results] == [r.attempt_scores for r in seq.results]


def test_one_errored_attempt_is_recorded_and_scores_zero():
    result = run_suite(DEMO, ALWAYS_CRASH_CEO, repeats=2)
    q002 = {r.case_id: r for r in result.results}["q002"]
    assert q002.attempt_scores == [0.0, 1.0]
    assert q002.score == 0.5
    assert q002.error == "1/2 attempts errored; first: target: RuntimeError: flaky upstream"


def test_repeats_must_be_positive():
    with pytest.raises(ValueError, match="repeats must be >= 1"):
        run_suite(DEMO, TARGET, repeats=0)


def _attempt(score, judge=None, cost=None, latency=10.0, error=None, answer="a"):
    return CaseResult(
        case_id="c",
        question="?",
        answer=answer,
        check_scores=[score],
        judge_score=judge,
        score=score,
        latency_ms=latency,
        error=error,
        judge_prompt_tokens=100 if cost is not None else None,
        judge_completion_tokens=10 if cost is not None else None,
        judge_cost_usd=cost,
    )


def test_aggregate_means_and_totals():
    merged = _aggregate(
        [
            _attempt(1.0, judge=1.0, cost=0.001, latency=10.0, answer="good"),
            _attempt(0.5, judge=0.5, cost=0.002, latency=30.0, answer="meh"),
        ]
    )
    assert merged.score == 0.75
    assert merged.judge_score == 0.75
    assert merged.latency_ms == 20.0
    assert merged.judge_cost_usd == pytest.approx(0.003)
    assert merged.judge_prompt_tokens == 200
    assert merged.answer == "meh"
    assert merged.error is None


# --- store --------------------------------------------------------------------


def test_store_round_trips_attempts_and_repeats(isolated_store):
    run_id = store.save_run(run_suite(DEMO, FLAKY, repeats=2))
    assert store.load_run_meta(run_id)["repeats"] == 2
    details = {d["case_id"]: d for d in store.load_case_details(run_id)}
    assert details["q001"]["attempt_scores"] == [1.0, 0.0]
    assert store.load_case_spreads(run_id)["q001"] == 1.0


def test_single_attempt_runs_have_no_spreads(isolated_store):
    run_id = store.save_run(run_suite(DEMO, TARGET))
    assert store.load_case_spreads(run_id) == {}


# --- CLI ----------------------------------------------------------------------


def test_cli_repeats_shows_spread_and_noisy_cases(isolated_store):
    res = CliRunner().invoke(
        app,
        ["run", "--suite", DEMO, "--target", FLAKY, "--repeats", "2", "--no-save"],
        env={"COLUMNS": "200"},
    )
    assert res.exit_code == 0, res.output
    out = _flat(res.output)
    assert "spread" in out
    assert "over 6 cases × 2 attempts" in out
    assert "Noisy:" in out
    assert "q001 (1.00)" in out


def test_cli_single_attempt_has_no_spread_column(isolated_store):
    res = CliRunner().invoke(app, ["run", "--suite", DEMO, "--target", TARGET, "--no-save"])
    assert "spread" not in res.output
    assert "Noisy" not in res.output
    assert "attempts" not in res.output


def test_cli_rejects_zero_repeats(isolated_store):
    res = CliRunner().invoke(
        app, ["run", "--suite", DEMO, "--target", TARGET, "--repeats", "0", "--no-save"]
    )
    assert res.exit_code == 2


def test_regression_on_noisy_case_is_flagged(isolated_store):
    # Baseline: deterministic demo, q001 = 1.0. New run: flaky, q001 mean 0.5
    # with spread 1.0 -> regressed past 0.2, and flagged as noisy.
    store.save_run(run_suite(DEMO, TARGET, repeats=2))
    store.pin_baseline("stable", "latest")
    res = CliRunner().invoke(
        app,
        ["run", "--suite", DEMO, "--target", FLAKY, "--repeats", "2", "--baseline", "stable",
         "--no-save"],
        env={"COLUMNS": "250"},
    )  # fmt: skip
    assert res.exit_code == 1  # the gate itself is unchanged
    out = _flat(res.output)
    assert "REGRESSED q001: 1.00 -> 0.50 (drop 0.50)" in out
    assert "(noisy: this run spread 1.00 — may be model noise)" in out


def test_regression_on_stable_case_is_not_flagged(isolated_store):
    from tests.test_error_visibility import _result  # 3 cases, q0..q2, all 1.0

    store.save_run(_result())
    store.pin_baseline("stable", "latest")
    report = store.regression_against_baseline("stable", {"q0": 0.0}, threshold=0.2)
    assert [c.case_id for c in report.regressed] == ["q0"]
    from evalx.cli import _noise_note

    assert _noise_note("q0", {}, {}, 0.2) == ""
    assert "baseline spread 0.50" in _noise_note("q0", {"q0": 0.5}, {}, 0.2)


# --- dashboard ----------------------------------------------------------------


def test_dashboard_runs_frame_has_repeats(isolated_store):
    store.save_run(run_suite(DEMO, TARGET, repeats=2))
    df = dashboard.runs_frame()
    assert df.iloc[0]["repeats"] == 2
