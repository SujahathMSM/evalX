"""End-to-end: the real ``evalx`` CLI against a live (fake) LLM server.

Every layer runs for real -- process, argument parsing, YAML loading, the
OpenAI-compatible adapter over HTTP, LiteLLM judge calls over HTTP, scoring,
the DuckDB store, baselines and the exit-code gates.
"""

import json
from pathlib import Path

import pytest

from evalx import store

pytestmark = pytest.mark.e2e

FRANCE = "What is the capital of France?"
MATH = "What is 2 + 2?"
SKY = "Why is the sky blue?"
PHOTO = "Summarise photosynthesis in one sentence."

SUITE = f"""
- id: e1
  question: "{FRANCE}"
  checks:
    - {{type: contains, expected: "Paris"}}
- id: e2
  question: "{MATH}"
  checks:
    - {{type: regex, expected: "\\\\b4\\\\b"}}
- id: e3
  question: "{SKY}"
  checks:
    - {{type: contains, expected: "scatter"}}
  judge: {{rubric: "Mentions Rayleigh scattering."}}
- id: e4
  question: "{PHOTO}"
  judge: {{rubric: "Says plants turn light into chemical energy."}}
"""

GOOD_ANSWERS = {
    FRANCE: "The capital of France is Paris.",
    MATH: "2 + 2 = 4.",
    SKY: "Because of Rayleigh scattering of sunlight.",
    PHOTO: "Plants turn light into chemical energy.",
}


@pytest.fixture
def suite(tmp_path) -> Path:
    path = tmp_path / "suite" / "cases.yaml"
    path.parent.mkdir()
    path.write_text(SUITE)
    return path


def _keyword_judge(prompt: str) -> str:
    """A judge that passes answers mentioning the rubric's key idea."""
    answer = prompt.split("Answer to score:\n", 1)[1]
    passed = "Rayleigh" in answer or "chemical energy" in answer
    return json.dumps({"score": 1.0 if passed else 0.0, "reason": "keyword check"})


TARGET = "examples.openai_compatible_app:answer"
JUDGE = "openai/fake-judge"


def _store(e2e_env) -> Path:
    return Path(e2e_env["EVALX_STORE"])


def test_judged_run_end_to_end(evalx_cli, fake_openai, suite, e2e_env):
    fake_openai.answers = dict(GOOD_ANSWERS)
    fake_openai.judge = _keyword_judge

    res = evalx_cli("run", "--suite", str(suite), "--target", TARGET, "--judge-model", JUDGE)

    assert res.returncode == 0, res.stdout + res.stderr
    assert "Run saved:" in res.stdout
    assert "Mean score: 1.000 over 4 cases" in res.stdout
    assert "Gate passed." in res.stdout

    # Real HTTP traffic: four app calls, two judge calls (only rubric cases).
    assert len(fake_openai.target_calls()) == 4
    assert all(c["model"] == "fake-target" for c in fake_openai.target_calls())
    judge_calls = fake_openai.judge_calls()
    assert len(judge_calls) == 2
    assert all(c["model"] == "fake-judge" for c in judge_calls)
    assert all(c["response_format"] == {"type": "json_object"} for c in judge_calls)
    assert all("temperature" not in c for c in judge_calls)  # omitted by default

    # Persisted: the store holds the run, the judge's verdicts and reasons.
    (run,) = store.list_runs(_store(e2e_env))
    assert run["judge_model"] == JUDGE
    assert (run["n_cases"], run["n_errors"], run["n_skipped"]) == (4, 0, 0)
    details = {d["case_id"]: d for d in store.load_case_details(run["run_id"], _store(e2e_env))}
    assert details["e3"]["judge_score"] == 1.0
    assert details["e3"]["judge_reason"] == "keyword check"
    assert details["e4"]["checks"] == []
    assert details["e1"]["answer"] == GOOD_ANSWERS[FRANCE]


def test_judge_scores_blend_with_code_checks(evalx_cli, fake_openai, suite, e2e_env):
    # e3 passes its code check ("scatter") but fails the rubric (no Rayleigh):
    # one check (weight 1) + judge (weight 1) -> 0.5.
    fake_openai.answers = dict(GOOD_ANSWERS, **{SKY: "Light scatters in the air."})
    fake_openai.judge = _keyword_judge

    res = evalx_cli("run", "--suite", str(suite), "--target", TARGET, "--judge-model", JUDGE)

    assert res.returncode == 0, res.stdout + res.stderr
    (run,) = store.list_runs(_store(e2e_env))
    scores = store.load_case_scores(run["run_id"], _store(e2e_env))
    assert scores["e3"] == pytest.approx(0.5)
    assert run["mean_score"] == pytest.approx(3.5 / 4)


def test_unjudged_run_skips_judge_only_case(evalx_cli, fake_openai, suite, e2e_env):
    fake_openai.answers = dict(GOOD_ANSWERS)

    res = evalx_cli("run", "--suite", str(suite), "--target", TARGET)

    assert res.returncode == 0, res.stdout + res.stderr
    assert "Skipped 1 judge-only case(s) (no --judge-model): e4" in res.stdout
    assert "Mean score: 1.000 over 3 cases" in res.stdout
    assert fake_openai.judge_calls() == []
    assert len(fake_openai.target_calls()) == 3  # e4 never reaches the app
    (run,) = store.list_runs(_store(e2e_env))
    assert (run["n_errors"], run["n_skipped"]) == (0, 1)


def test_target_outage_trips_infra_gate_and_blocks_pinning(
    evalx_cli, fake_openai, suite, e2e_env
):
    fake_openai.answers = dict(GOOD_ANSWERS)
    fake_openai.failures = {MATH: (429, {"error": {"message": "rate limit exceeded"}})}

    res = evalx_cli("run", "--suite", str(suite), "--target", TARGET)

    assert res.returncode == 1
    assert "ERROR e2: target: OpenAICompatError:" in res.stdout
    assert "HTTP 429: rate limit exceeded" in res.stdout
    assert "INFRA FAIL: 1/3 cases errored (rate 0.33 > allowed 0.00)" in res.stdout
    assert res.stdout.index("INFRA FAIL") < res.stdout.index("Gate passed.")

    # The outage run was still saved, with its error count and error text.
    (run,) = store.list_runs(_store(e2e_env))
    assert run["n_errors"] == 1
    details = {d["case_id"]: d for d in store.load_case_details(run["run_id"], _store(e2e_env))}
    assert details["e2"]["score"] == 0.0
    assert "HTTP 429" in details["e2"]["error"]

    # `evalx runs` makes it visible; pinning it as a baseline is refused.
    runs = evalx_cli("runs")
    assert runs.returncode == 0
    row = next(line for line in runs.stdout.splitlines() if run["run_id"] in line)
    cells = [c.strip() for c in row.split("│") if c.strip()]
    assert cells[4:6] == ["1", "1"]  # Errors, Skipped

    pin = evalx_cli("baseline", "pin", "latest")
    assert pin.returncode == 1
    assert "1 error case(s)" in pin.stdout
    assert "--allow-errors" in pin.stdout
    assert evalx_cli("baseline", "pin", "latest", "--allow-errors").returncode == 0


def test_outage_passes_infra_gate_when_allowed(evalx_cli, fake_openai, suite):
    fake_openai.answers = dict(GOOD_ANSWERS)
    fake_openai.failures = {MATH: (500, {"error": {"message": "boom"}})}

    res = evalx_cli(
        "run", "--suite", str(suite), "--target", TARGET, "--no-save", "--max-error-rate", "0.5"
    )

    assert res.returncode == 0, res.stdout + res.stderr
    assert "INFRA FAIL" not in res.stdout
    assert "Run saved" not in res.stdout


def test_malformed_judge_verdict_is_recorded(evalx_cli, fake_openai, suite, e2e_env):
    fake_openai.answers = dict(GOOD_ANSWERS)
    fake_openai.judge = lambda prompt: "certainly! the answer looks great"

    res = evalx_cli(
        "run", "--suite", str(suite), "--target", TARGET, "--judge-model", JUDGE,
        "--max-error-rate", "1.0",
    )  # fmt: skip

    assert res.returncode == 0, res.stdout + res.stderr
    assert "ERROR e3: judge: JudgeError: judge returned invalid verdict" in res.stdout
    # Retry once, then give up: two judge calls per rubric case.
    sky_calls = [c for c in fake_openai.judge_calls() if SKY in c["messages"][-1]["content"]]
    assert len(sky_calls) == 2
    (run,) = store.list_runs(_store(e2e_env))
    assert run["n_errors"] == 2  # e3 and e4 both have rubrics
    scores = store.load_case_scores(run["run_id"], _store(e2e_env))
    assert scores["e3"] == 0.0  # no code-only fallback when the judge fails


def test_baseline_regression_gate_end_to_end(evalx_cli, fake_openai, suite):
    fake_openai.answers = dict(GOOD_ANSWERS)
    assert evalx_cli("run", "--suite", str(suite), "--target", TARGET).returncode == 0
    pin = evalx_cli("baseline", "pin", "latest", "--name", "stable")
    assert pin.returncode == 0, pin.stdout
    assert "pinned to run" in pin.stdout

    # The app regresses on one case while the others stay perfect.
    fake_openai.answers[FRANCE] = "The capital of France is Lyon."
    res = evalx_cli("run", "--suite", str(suite), "--target", TARGET, "--baseline", "stable")

    assert res.returncode == 1
    assert "REGRESSED e1: 1.00 -> 0.00 (drop 1.00)" in res.stdout
    assert "REGRESSION GATE FAILED: 1 case(s)" in res.stdout

    cmp = evalx_cli("compare", "latest-1", "latest")
    assert cmp.returncode == 0, cmp.stdout + cmp.stderr
    assert "e1" in cmp.stdout


def test_unjudged_run_against_judged_baseline_notes_skipped(evalx_cli, fake_openai, suite):
    fake_openai.answers = dict(GOOD_ANSWERS)
    fake_openai.judge = _keyword_judge
    judged = evalx_cli("run", "--suite", str(suite), "--target", TARGET, "--judge-model", JUDGE)
    assert judged.returncode == 0, judged.stdout
    assert evalx_cli("baseline", "pin", "latest").returncode == 0

    res = evalx_cli("run", "--suite", str(suite), "--target", TARGET, "--baseline", "stable")

    assert "e4 is in the baseline but not in this run" in res.stdout
    assert (
        "note: baseline was judged, this run is unjudged: 1 case(s) that have judge scores "
        "in the baseline were not compared"
    ) in res.stdout


def test_unreachable_target_is_an_error_not_a_crash(evalx_cli, suite, e2e_env):
    e2e_env["OPENAI_COMPAT_BASE_URL"] = "http://127.0.0.1:9/v1"  # discard port: refused

    res = evalx_cli("run", "--suite", str(suite), "--target", TARGET, "--no-save")

    assert res.returncode == 1
    assert "INFRA FAIL: 3/3 cases errored" in res.stdout
    assert "cannot reach an OpenAI-compatible server" in res.stdout
    assert "Traceback" not in res.stderr
