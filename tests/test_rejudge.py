"""evalx rejudge: re-grade stored answers with the judge only."""

import threading
from collections import Counter
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalx import store
from evalx.cli import app
from evalx.judge import JudgeVerdict
from evalx.noise import noise_report
from evalx.rejudge import RejudgeError, rejudge_run
from evalx.runner import run_suite

DEMO = "datasets/demo/v1"
COUNTED = "tests.test_rejudge:counted_answer"
RUBRIC_CASES = ["q003", "q006"]  # the demo cases with a judge rubric

_target_calls: Counter = Counter()


def counted_answer(question: str) -> str:
    """The demo app, counting calls so tests can prove rejudge never calls it."""
    from examples.demo_app import answer

    _target_calls[question] += 1
    return answer(question)


class SequenceJudge:
    """Returns the next score from a fixed sequence, per case, in call order."""

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls: Counter = Counter()
        self.lock = threading.Lock()

    def score(self, case, answer):
        with self.lock:
            i = self.calls[case.id]
            self.calls[case.id] += 1
        return JudgeVerdict(score=self.scores[i % len(self.scores)], reason=f"call {i}")


class FixedJudge(SequenceJudge):
    def __init__(self, score):
        super().__init__([score])


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALX_STORE", str(tmp_path / "runs.duckdb"))
    _target_calls.clear()


def _flat(text: str) -> str:
    return " ".join(text.split())


def _judged_source(repeats: int = 1) -> str:
    return store.save_run(
        run_suite(DEMO, COUNTED, judge=FixedJudge(1.0), repeats=repeats), judge_model="m"
    )


# --- core ----------------------------------------------------------------------


def test_rejudge_measures_judge_only_spread(isolated_store):
    source = _judged_source()
    calls_before = sum(_target_calls.values())
    judge = SequenceJudge([0.5, 1.0, 1.0])
    out = rejudge_run(source, judge, repeats=3)

    assert sum(_target_calls.values()) == calls_before  # the app is never called
    assert dict(judge.calls) == {"q003": 3, "q006": 3}
    by_id = {c.case_id: c for c in out.rejudged}
    assert set(by_id) == set(RUBRIC_CASES)
    assert by_id["q003"].judge_scores == [0.5, 1.0, 1.0]
    assert by_id["q003"].spread == 0.5
    assert by_id["q003"].original_judge == 1.0


def test_stored_checks_are_reused_and_scores_blend_as_in_a_run(isolated_store):
    source = _judged_source()
    out = rejudge_run(source, FixedJudge(1.0), repeats=2)
    original = {d["case_id"]: d for d in store.load_case_details(source)}
    regraded = {r.case_id: r for r in out.result.results}
    # Same answers, same checks, same judge score -> identical case scores.
    for case_id, d in original.items():
        assert regraded[case_id].score == pytest.approx(d["score"])
        assert regraded[case_id].answer == d["answer"]
    assert regraded["q003"].attempt_scores == pytest.approx([2 / 3, 2 / 3])


def test_cases_without_rubric_are_carried_over(isolated_store):
    source = _judged_source()
    out = rejudge_run(source, FixedJudge(0.0), repeats=1)
    assert [r.case_id for r in out.result.results] == [
        d["case_id"] for d in store.load_case_details(source)
    ]
    reasons = {s.case_id: s.reason for s in out.skipped}
    assert reasons["q001"] == "no rubric"
    q001 = {r.case_id: r for r in out.result.results}["q001"]
    assert q001.score == 1.0  # unchanged


def test_parallel_rejudge_matches_sequential(isolated_store):
    source = _judged_source()
    seq = rejudge_run(source, FixedJudge(0.5), repeats=3)
    par = rejudge_run(source, FixedJudge(0.5), repeats=3, workers=4)
    assert [r.score for r in par.result.results] == [r.score for r in seq.result.results]


def test_other_judge_model_grades_same_answers(isolated_store):
    source = _judged_source()
    harsh = rejudge_run(source, FixedJudge(0.0), repeats=1, judge_model="other")
    assert harsh.judge_model == "other"
    q003 = {r.case_id: r for r in harsh.result.results}["q003"]
    # code 0.5 over weight 2 + judge 0.0 over weight 1 = 1/3
    assert q003.score == pytest.approx(1 / 3)


def test_errored_and_removed_cases_are_skipped(isolated_store, tmp_path):
    from tests.test_runner import _crash_on_ceo  # noqa: F401  (target below)

    source = store.save_run(
        run_suite(DEMO, "tests.test_runner:_crash_on_ceo", judge=FixedJudge(1.0)),
        judge_model="m",
    )
    # A copy of the dataset without q006.
    trimmed = tmp_path / "trimmed.yaml"
    lines = Path(DEMO, "cases.yaml").read_text(encoding="utf-8").split("\n- id: ")
    trimmed.write_text("\n- id: ".join(p for p in lines if not p.startswith("q006")))
    out = rejudge_run(source, FixedJudge(1.0), repeats=1, suite=trimmed)
    reasons = {s.case_id: s.reason for s in out.skipped}
    assert reasons["q002"] == "errored in the original run"
    assert reasons["q006"] == "no longer in the dataset"
    assert [c.case_id for c in out.rejudged] == ["q003"]


def test_changed_question_is_reported(isolated_store, tmp_path):
    source = _judged_source()
    edited = tmp_path / "edited.yaml"
    text = Path(DEMO, "cases.yaml").read_text(encoding="utf-8")
    edited.write_text(text.replace("Why did operating costs rise in Q3?", "Why did costs rise?"))
    out = rejudge_run(source, FixedJudge(1.0), repeats=1, suite=edited)
    assert out.question_changed == ["q003"]


def test_missing_dataset_says_how_to_fix_it(isolated_store, tmp_path):
    source = _judged_source()
    with pytest.raises(RejudgeError, match="pass --suite"):
        rejudge_run(source, FixedJudge(1.0), suite=tmp_path / "nope")


def test_nothing_to_rejudge(isolated_store):
    source = store.save_run(run_suite(DEMO, COUNTED))  # unjudged: q003/q006 still have rubrics
    out = rejudge_run(source, FixedJudge(1.0), repeats=1)
    assert [c.case_id for c in out.rejudged] == RUBRIC_CASES
    assert all(c.original_judge is None for c in out.rejudged)


def test_saved_rejudge_links_to_source_and_noise_reads_it(isolated_store):
    source = _judged_source()
    out = rejudge_run(source, SequenceJudge([0.0, 1.0, 1.0]), repeats=3)
    new_id = store.save_run(out.result, "m", rejudge_of=source)
    assert store.load_run_meta(new_id)["rejudge_of"] == source
    rep = noise_report(new_id)  # judge-only wobble
    assert rep.spreads["q003"] == pytest.approx(1 / 3)  # judge 0 vs 1 at weight 1 of 3
    # Only re-graded cases have several judge attempts; carried-over cases
    # (no rubric) were not measured, so noise leaves them out.
    assert set(rep.spreads) == set(RUBRIC_CASES)


# --- CLI -----------------------------------------------------------------------


class _Resp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]
        self.usage = type("U", (), {"prompt_tokens": 100, "completion_tokens": 10})()


@pytest.fixture
def litellm_judge(monkeypatch):
    """LLMJudge with LiteLLM stubbed: scores cycle 1.0, 0.5, 1.0 per call."""
    seq = iter([1.0, 0.5, 1.0] * 100)
    monkeypatch.setattr(
        "evalx.judge.litellm.completion",
        lambda **kw: _Resp(f'{{"score": {next(seq)}, "reason": "ok"}}'),
    )
    monkeypatch.setattr("evalx.judge.litellm.completion_cost", lambda completion_response: 0.001)


def test_cli_rejudge(isolated_store, litellm_judge):
    source = _judged_source()
    res = CliRunner().invoke(app, ["rejudge", source, "--repeats", "3"], env={"COLUMNS": "200"})
    assert res.exit_code == 0, res.output
    out = _flat(res.output)
    assert f"Re-grading stored answers of {source} with m" in out
    assert "2 case(s) with a rubric × 3 · app not called" in out
    assert "Judge noise: 90% of cases move at most 0.50" in out
    assert "skipped 4 (no rubric)" in out
    assert "Judge usage: 660 tokens, $0.0060" in out
    assert f"(rejudge of {source})" in out
    (latest,) = [r for r in store.list_runs() if r["rejudge_of"] == source]
    assert latest["repeats"] == 3


def test_cli_rejudge_unjudged_run_needs_a_model(isolated_store):
    source = store.save_run(run_suite(DEMO, COUNTED))
    res = CliRunner().invoke(app, ["rejudge", source], env={"COLUMNS": "200"})
    assert res.exit_code == 2
    assert "pass --judge-model" in _flat(res.output)


def test_cli_rejudge_notes_repeats_source(isolated_store, litellm_judge):
    source = _judged_source(repeats=2)
    res = CliRunner().invoke(
        app, ["rejudge", source, "--repeats", "2", "--no-save"], env={"COLUMNS": "200"}
    )
    assert res.exit_code == 0, res.output
    out = _flat(res.output)
    assert "used --repeats 2" in out
    assert "Compare the spreads, not the two judge columns" in out
    assert "Run saved" not in out


def test_cli_rejudge_unknown_run(isolated_store):
    res = CliRunner().invoke(app, ["rejudge", "nope"])
    assert res.exit_code == 1
    assert "no runs stored yet" in res.output


def test_cli_compare_names_the_rejudge(isolated_store):
    source = _judged_source()
    out = rejudge_run(source, FixedJudge(0.5), repeats=1)
    new_id = store.save_run(out.result, "m", rejudge_of=source)
    res = CliRunner().invoke(app, ["compare", source, new_id], env={"COLUMNS": "200"})
    assert res.exit_code == 0, res.output
    assert f"{new_id} is a rejudge of {source}" in _flat(res.output)
    # Judge 1.0 -> 0.5 on the rubric cases moves only those two scores.
    comp = store.compare_runs(source, new_id)
    assert comp.regressed == 2 and comp.unchanged == 4
