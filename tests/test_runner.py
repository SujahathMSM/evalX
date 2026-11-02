"""Tests for the runner: the harness must catch the demo app's failures."""

import pytest

from evalx.adapters import AdapterError, resolve_target
from evalx.runner import run_suite

DEMO = "datasets/demo/v1"
TARGET = "examples.demo_app:answer"


def test_run_demo_suite():
    result = run_suite(DEMO, TARGET)
    assert result.n_cases == 6
    assert result.suite == DEMO
    assert result.target == TARGET
    scores = {r.case_id: r.score for r in result.results}
    # Easy cases pass fully.
    assert scores["q001"] == 1.0
    assert scores["q002"] == 1.0
    # Medium cases get partial credit (one of two checks each).
    assert scores["q003"] == 0.5
    assert scores["q006"] == 0.5
    # Adversarial cases: the demo app hallucinates, scores 0.
    assert scores["q004"] == 0.0
    assert scores["q005"] == 0.0
    # Mean = (1 + 1 + 0.5 + 0 + 0 + 0.5) / 6 = 0.5
    assert result.mean_score == 0.5


def test_latency_recorded():
    result = run_suite(DEMO, TARGET)
    assert all(r.latency_ms >= 0 for r in result.results)


def test_bad_target_raises():
    with pytest.raises(AdapterError):
        resolve_target("no-colon-here")


def test_missing_module_raises():
    with pytest.raises(AdapterError):
        resolve_target("no_such_module:fn")


def test_non_callable_raises():
    with pytest.raises(AdapterError):
        resolve_target("evalx:__version__")  # a string, not a q->a function


class StubJudge:
    """Test double for the LLM judge: returns a fixed score."""

    def __init__(self, score: float):
        self._score = score

    def score(self, case, answer):
        from evalx.judge import JudgeVerdict

        return JudgeVerdict(score=self._score, reason="stub")


def test_judge_combines_with_code_checks():
    # q003: code 0.5 over weight 2 (two checks) + judge 1.0 over weight 1
    #   -> (0.5*2 + 1.0*1) / 3 = 0.666...
    # q001: no judge rubric -> judge skipped, score stays 1.0
    result = run_suite(DEMO, TARGET, judge=StubJudge(1.0))
    scores = {r.case_id: r for r in result.results}
    assert scores["q003"].score == pytest.approx(2.0 / 3.0)
    assert scores["q003"].judge_score == 1.0
    assert scores["q003"].judge_reason == "stub"
    assert scores["q001"].score == 1.0
    assert scores["q001"].judge_score is None


def test_no_judge_means_no_judge_scores():
    result = run_suite(DEMO, TARGET)
    assert all(r.judge_score is None for r in result.results)
    assert result.mean_score == 0.5  # unchanged from M1


# --- Robustness: one bad case must not sink the run -------------------------

def _crash_on_ceo(question: str) -> str:
    """Target that raises on one question and defers to the demo app otherwise."""
    from examples.demo_app import answer

    if "CEO" in question:
        raise RuntimeError("upstream 500")
    return answer(question)


def test_target_exception_is_recorded_not_raised():
    result = run_suite(DEMO, "tests.test_runner:_crash_on_ceo")
    assert result.n_cases == 6
    by_id = {r.case_id: r for r in result.results}
    assert by_id["q002"].score == 0.0
    assert by_id["q002"].answer == ""
    assert by_id["q002"].error == "target: RuntimeError: upstream 500"
    # The other cases still ran and scored normally.
    assert by_id["q001"].score == 1.0
    assert by_id["q001"].error is None
    # q002 counts as 0 in the mean: (1 + 0 + 0.5 + 0 + 0 + 0.5) / 6
    assert result.mean_score == pytest.approx(2.0 / 6.0)


class BrokenJudge:
    def score(self, case, answer):
        raise TimeoutError("judge timed out")


def test_judge_exception_is_recorded_and_case_scores_zero():
    result = run_suite(DEMO, TARGET, judge=BrokenJudge())
    by_id = {r.case_id: r for r in result.results}
    # q003 has a rubric: the judge failure is recorded, the case scores 0.
    assert by_id["q003"].error == "judge: TimeoutError: judge timed out"
    assert by_id["q003"].score == 0.0
    assert by_id["q003"].judge_score is None
    # Code checks are still kept for the record.
    assert sorted(by_id["q003"].check_scores) == [0.0, 1.0]
    # Cases without a rubric never touch the judge.
    assert by_id["q001"].error is None
    assert by_id["q001"].score == 1.0


def _write_suite(tmp_path, body: str):
    path = tmp_path / "cases.yaml"
    path.write_text(body)
    return path


JUDGE_ONLY_SUITE = """
- id: c1
  question: "What was the company's revenue in 2024?"
  checks:
    - {type: contains, expected: "$4.2M"}
- id: c2
  question: "Summarise the year."
  judge: {rubric: "Mentions revenue growth."}
"""


def test_judge_only_case_skipped_without_judge(tmp_path):
    suite = _write_suite(tmp_path, JUDGE_ONLY_SUITE)
    result = run_suite(suite, TARGET)
    assert result.skipped == ["c2"]
    assert [r.case_id for r in result.results] == ["c1"]
    assert result.n_cases == 1
    # c2 is left out of the mean instead of dragging it to 0.5.
    assert result.mean_score == 1.0


def test_judge_only_case_scored_with_judge(tmp_path):
    suite = _write_suite(tmp_path, JUDGE_ONLY_SUITE)
    result = run_suite(suite, TARGET, judge=StubJudge(0.4))
    assert result.skipped == []
    by_id = {r.case_id: r for r in result.results}
    assert by_id["c2"].score == pytest.approx(0.4)
    assert result.mean_score == pytest.approx(0.7)
