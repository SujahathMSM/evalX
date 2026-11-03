"""Judge token and cost tracking: judge -> runner -> store -> CLI/dashboard.

LiteLLM is stubbed: completion() returns canned responses with a usage
block, and completion_cost() returns a price (or raises, like LiteLLM does
for a model it has no price for).
"""

import duckdb
import pytest
from typer.testing import CliRunner

from evalx import dashboard, store
from evalx.cli import app
from evalx.judge import JudgeError, JudgeUsage, LLMJudge
from evalx.runner import run_suite
from evalx.schemas import CaseResult, JudgeSpec, RunResult, TestCase

from .helpers import table_cells

DEMO = "datasets/demo/v1"
TARGET = "examples.demo_app:answer"
HAIKU = "anthropic/claude-haiku-5-5"


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALX_STORE", str(tmp_path / "runs.duckdb"))


def _flat(text: str) -> str:
    return " ".join(text.split())


class _Usage:
    def __init__(self, prompt: int, completion: int):
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _Response:
    def __init__(self, content: str, prompt: int = 100, completion: int = 10):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]
        self.usage = _Usage(prompt, completion)


def _priced(monkeypatch, per_call: float = 0.001):
    """completion_cost: a flat price per response, like a known model."""
    monkeypatch.setattr(
        "evalx.judge.litellm.completion_cost", lambda completion_response: per_call
    )


def _unpriced(monkeypatch):
    def raise_unmapped(completion_response):
        raise RuntimeError("This model isn't mapped yet.")  # like ModelNotMappedError

    monkeypatch.setattr("evalx.judge.litellm.completion_cost", raise_unmapped)


def _replies(monkeypatch, *contents: str):
    queue = list(contents)
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return _Response(queue.pop(0) if len(queue) > 1 else queue[0])

    monkeypatch.setattr("evalx.judge.litellm.completion", completion)
    return calls


CASE = TestCase(id="q", question="Why?", judge=JudgeSpec(rubric="Says because."))
GOOD = '{"score": 1.0, "reason": "ok"}'


# --- JudgeUsage arithmetic ----------------------------------------------------


def test_usage_add_sums_known_values():
    total = JudgeUsage(prompt_tokens=1, completion_tokens=2, cost_usd=0.5) + JudgeUsage(
        prompt_tokens=3, completion_tokens=4, cost_usd=0.25
    )
    assert (total.prompt_tokens, total.completion_tokens, total.cost_usd) == (4, 6, 0.75)


def test_usage_add_unknown_stays_unknown():
    total = JudgeUsage(prompt_tokens=1, completion_tokens=2, cost_usd=0.5) + JudgeUsage(
        prompt_tokens=3, completion_tokens=4, cost_usd=None
    )
    assert total.cost_usd is None  # never report a partial price as the total
    assert total.prompt_tokens == 4


# --- LLMJudge captures usage --------------------------------------------------


def test_judge_verdict_carries_usage(monkeypatch):
    _replies(monkeypatch, GOOD)
    _priced(monkeypatch, 0.002)
    verdict = LLMJudge(HAIKU).score(CASE, "because")
    assert verdict.score == 1.0
    assert verdict.usage == JudgeUsage(prompt_tokens=100, completion_tokens=10, cost_usd=0.002)


def test_judge_usage_sums_retries(monkeypatch):
    calls = _replies(monkeypatch, "not json", GOOD)
    _priced(monkeypatch, 0.002)
    verdict = LLMJudge(HAIKU).score(CASE, "because")
    assert len(calls) == 2
    assert verdict.usage.prompt_tokens == 200  # the failed attempt was paid for too
    assert verdict.usage.cost_usd == pytest.approx(0.004)


def test_judge_error_carries_spent_usage(monkeypatch):
    _replies(monkeypatch, "not json")
    _priced(monkeypatch, 0.002)
    with pytest.raises(JudgeError) as exc_info:
        LLMJudge(HAIKU).score(CASE, "because")
    assert exc_info.value.usage.prompt_tokens == 200
    assert exc_info.value.usage.cost_usd == pytest.approx(0.004)


def test_unpriced_model_keeps_tokens_drops_cost(monkeypatch):
    _replies(monkeypatch, GOOD)
    _unpriced(monkeypatch)
    verdict = LLMJudge("openai/my-local-model").score(CASE, "because")
    assert verdict.usage.prompt_tokens == 100
    assert verdict.usage.cost_usd is None


def test_model_cannot_report_its_own_usage(monkeypatch):
    # The judge's JSON is untrusted: a "usage" key in it must be ignored.
    _replies(
        monkeypatch,
        '{"score": 1.0, "reason": "ok", "usage": {"prompt_tokens": 0, "cost_usd": 0.0}}',
    )
    _priced(monkeypatch, 0.002)
    verdict = LLMJudge(HAIKU).score(CASE, "because")
    assert verdict.usage.cost_usd == 0.002
    assert verdict.usage.prompt_tokens == 100


def test_non_object_verdict_is_retried_then_raises(monkeypatch):
    calls = _replies(monkeypatch, "[1, 2]")
    _priced(monkeypatch)
    with pytest.raises(JudgeError, match="not a JSON object"):
        LLMJudge(HAIKU).score(CASE, "because")
    assert len(calls) == 2


# --- Runner and store ---------------------------------------------------------


def test_runner_records_judge_usage_per_case(monkeypatch):
    _replies(monkeypatch, GOOD)
    _priced(monkeypatch, 0.001)
    result = run_suite(DEMO, TARGET, judge=LLMJudge(HAIKU))
    by_id = {r.case_id: r for r in result.results}
    # Demo rubric cases: q003, q006. Others never call the judge.
    assert by_id["q003"].judge_cost_usd == 0.001
    assert by_id["q003"].judge_prompt_tokens == 100
    assert by_id["q003"].judge_completion_tokens == 10
    assert by_id["q001"].judge_cost_usd is None
    assert store.run_judge_usage(result) == (220, pytest.approx(0.002))


def test_runner_records_usage_of_failed_judge(monkeypatch):
    _replies(monkeypatch, "not json")
    _priced(monkeypatch, 0.001)
    result = run_suite(DEMO, TARGET, judge=LLMJudge(HAIKU))
    q003 = {r.case_id: r for r in result.results}["q003"]
    assert q003.error.startswith("judge: JudgeError")
    assert q003.judge_cost_usd == pytest.approx(0.002)  # two paid attempts


def test_unjudged_run_has_no_usage():
    result = run_suite(DEMO, TARGET)
    assert store.run_judge_usage(result) == (None, None)


def _result(costs: list[float | None], latency: float = 10.0, prefix: str = "q") -> RunResult:
    results = [
        CaseResult(
            case_id=f"{prefix}{i}",
            question="?",
            answer="a",
            check_scores=[],
            judge_score=1.0 if cost is not None else None,
            score=1.0,
            latency_ms=latency,
            judge_prompt_tokens=100 if cost is not None else None,
            judge_completion_tokens=10 if cost is not None else None,
            judge_cost_usd=cost,
        )
        for i, cost in enumerate(costs)
    ]
    return RunResult(suite="s", target="t", n_cases=len(results), mean_score=1.0, results=results)


def test_store_round_trips_usage(isolated_store):
    run_id = store.save_run(_result([0.001, None, 0.003]), judge_model=HAIKU)
    meta = store.load_run_meta(run_id)
    assert meta["judge_tokens"] == 220
    assert meta["judge_cost_usd"] == pytest.approx(0.004)
    details = {d["case_id"]: d for d in store.load_case_details(run_id)}
    assert details["q0"]["judge_cost_usd"] == 0.001
    assert details["q0"]["judge_prompt_tokens"] == 100
    assert details["q1"]["judge_cost_usd"] is None


def test_unjudged_run_stores_null_usage(isolated_store):
    run_id = store.save_run(_result([None, None]))
    meta = store.load_run_meta(run_id)
    assert meta["judge_tokens"] is None
    assert meta["judge_cost_usd"] is None


def test_legacy_store_gains_usage_columns(isolated_store):
    path = store.store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE runs (run_id VARCHAR PRIMARY KEY, suite VARCHAR, target VARCHAR,"
        " judge_model VARCHAR, mean_score DOUBLE, n_cases INTEGER,"
        " created_at TIMESTAMP DEFAULT now())"
    )
    con.execute(
        "CREATE TABLE case_results (run_id VARCHAR, case_id VARCHAR, score DOUBLE,"
        " judge_score DOUBLE, latency_ms DOUBLE)"
    )
    con.execute("INSERT INTO runs VALUES ('old-run', 's', 't', NULL, 0.5, 1, now())")
    con.execute("INSERT INTO case_results VALUES ('old-run', 'q0', 1.0, NULL, 1.0)")
    con.close()
    (old,) = store.list_runs()
    assert old["judge_cost_usd"] is None
    (detail,) = store.load_case_details("old-run")
    assert detail["judge_cost_usd"] is None
    new_id = store.save_run(_result([0.001]))
    assert store.load_run_meta(new_id)["judge_cost_usd"] == 0.001


# --- compare: latency and cost next to score ----------------------------------


def test_compare_reports_latency_and_cost(isolated_store):
    store.save_run(_result([0.002, 0.002], latency=100.0))
    store.save_run(_result([0.001, 0.001], latency=50.0))
    comp = store.compare_runs("latest-1", "latest")
    assert comp.mean_latency_a == 100.0
    assert comp.mean_latency_b == 50.0
    assert comp.judge_cost_a == pytest.approx(0.004)
    assert comp.judge_cost_b == pytest.approx(0.002)
    assert comp.cases[0].latency_a == 100.0
    assert comp.cases[0].cost_b == 0.001


def test_compare_uses_shared_cases_only(isolated_store):
    store.save_run(_result([0.002, 0.002, 0.5], latency=100.0))  # q2 not in B
    store.save_run(_result([0.001, 0.001], latency=50.0))
    comp = store.compare_runs("latest-1", "latest")
    assert comp.judge_cost_a == pytest.approx(0.004)  # the 0.5 case isn't shared


def test_compare_unpriced_runs_have_no_cost(isolated_store):
    store.save_run(_result([None]))
    store.save_run(_result([None]))
    comp = store.compare_runs("latest-1", "latest")
    assert comp.judge_cost_a is None and comp.judge_cost_b is None
    assert comp.mean_latency_a == 10.0


def test_cli_compare_prints_latency_and_cost(isolated_store):
    store.save_run(_result([0.002, 0.002], latency=100.0))
    store.save_run(_result([0.001, 0.001], latency=50.0))
    res = CliRunner().invoke(app, ["compare", "latest-1", "latest"])
    assert res.exit_code == 0
    out = _flat(res.output)
    assert "Latency (mean per case): 100 -> 50 ms (-50%)" in out
    assert "Judge cost: $0.0040 -> $0.0020 (-50%)" in out
    assert "judge $ a" in out


def test_cli_compare_hides_cost_when_unpriced(isolated_store):
    store.save_run(_result([None]))
    store.save_run(_result([None]))
    res = CliRunner().invoke(app, ["compare", "latest-1", "latest"])
    assert res.exit_code == 0
    assert "Latency (mean per case): 10 -> 10 ms (+0%)" in _flat(res.output)
    assert "Judge cost" not in res.output
    assert "judge $" not in res.output


# --- run and runs output ------------------------------------------------------


def test_cli_run_prints_judge_usage(isolated_store, monkeypatch):
    _replies(monkeypatch, GOOD)
    _priced(monkeypatch, 0.001)
    res = CliRunner().invoke(
        app, ["run", "--suite", DEMO, "--target", TARGET, "--judge-model", HAIKU]
    )
    assert res.exit_code == 0
    assert "Judge usage: 220 tokens, $0.0020" in _flat(res.output)
    (run,) = store.list_runs()
    assert run["judge_cost_usd"] == pytest.approx(0.002)


def test_cli_run_unpriced_judge_says_cost_unknown(isolated_store, monkeypatch):
    _replies(monkeypatch, GOOD)
    _unpriced(monkeypatch)
    res = CliRunner().invoke(
        app,
        ["run", "--suite", DEMO, "--target", TARGET, "--judge-model", "openai/local", "--no-save"],
    )
    assert res.exit_code == 0
    assert "Judge usage: 220 tokens, cost unknown (no LiteLLM price for openai/local)" in _flat(
        res.output
    )


def test_cli_run_without_judge_prints_no_usage(isolated_store):
    res = CliRunner().invoke(app, ["run", "--suite", DEMO, "--target", TARGET, "--no-save"])
    assert "Judge usage" not in res.output


def test_cli_runs_shows_judge_cost(isolated_store):
    store.save_run(_result([0.0015]))
    store.save_run(_result([None], prefix="x"))
    res = CliRunner().invoke(app, ["runs"], env={"COLUMNS": "200"})
    assert res.exit_code == 0
    assert "judge $" in res.output
    rows = [line for line in res.output.splitlines() if " s " in line]
    cells = [table_cells(r) for r in rows]
    assert sorted(c[6] for c in cells) == ["$0.0015", "–"]


# --- dashboard ----------------------------------------------------------------


def test_dashboard_frames_carry_latency_and_cost(isolated_store):
    store.save_run(_result([0.002], latency=100.0))
    store.save_run(_result([0.001], latency=50.0))
    runs = dashboard.runs_frame()
    assert "judge_cost_usd" in runs.columns
    df, summary = dashboard.comparison_frame("latest-1", "latest")
    row = df.iloc[0]
    assert (row["latency_a"], row["latency_b"]) == (100.0, 50.0)
    assert (row["cost_a"], row["cost_b"]) == (0.002, 0.001)
    assert summary["mean_latency_b"] == 50.0
    assert summary["judge_cost_a"] == 0.002


def test_dashboard_metric_formatters():
    assert dashboard._fmt_usd(None) == "–"
    assert dashboard._fmt_usd(0.0021) == "$0.0021"
    assert dashboard._fmt_ms(12.6) == "13"
    assert dashboard._fmt_delta(None, 1.0, "{:+.0f}") is None
    assert dashboard._fmt_delta(100.0, 50.0, "{:+.0f} ms") == "-50 ms"


# --- wall time and workers --------------------------------------------------


def test_run_records_duration_and_workers():
    seq = run_suite(DEMO, TARGET)
    par = run_suite(DEMO, TARGET, workers=3)
    assert seq.duration_s is not None and seq.duration_s > 0
    assert (seq.workers, par.workers) == (1, 3)


def _timed(duration: float, workers: int) -> RunResult:
    return _result([None]).model_copy(update={"duration_s": duration, "workers": workers})


def test_store_round_trips_duration_and_workers(isolated_store):
    run_id = store.save_run(_timed(236.1, 4))
    meta = store.load_run_meta(run_id)
    assert (meta["duration_s"], meta["workers"]) == (236.1, 4)


def test_compare_reports_wall_time_and_workers(isolated_store):
    store.save_run(_timed(264.0, 1))
    store.save_run(_timed(236.0, 4))
    comp = store.compare_runs("latest-1", "latest")
    assert (comp.duration_a, comp.duration_b) == (264.0, 236.0)
    assert (comp.workers_a, comp.workers_b) == (1, 4)


def test_cli_compare_explains_differing_workers(isolated_store):
    store.save_run(_timed(264.0, 1))
    store.save_run(_timed(236.0, 4))
    res = CliRunner().invoke(app, ["compare", "latest-1", "latest"])
    out = _flat(res.output)
    assert "Wall time (whole run): 264.0s -> 236.0s (-11%) | workers 1 -> 4" in out
    assert "the runs used different --workers" in out


def test_cli_compare_no_workers_note_when_equal(isolated_store):
    store.save_run(_timed(10.0, 2))
    store.save_run(_timed(12.0, 2))
    res = CliRunner().invoke(app, ["compare", "latest-1", "latest"])
    assert "Wall time (whole run): 10.0s -> 12.0s (+20%) | workers 2 -> 2" in _flat(res.output)
    assert "different --workers" not in res.output


def test_cli_compare_legacy_runs_show_unknown_wall_time(isolated_store):
    store.save_run(_result([None]))  # duration/workers not set: like an old run
    store.save_run(_timed(5.0, 1))
    res = CliRunner().invoke(app, ["compare", "latest-1", "latest"])
    out = _flat(res.output)
    assert "Wall time (whole run): – -> 5.0s | workers – -> 1" in out
    assert "different --workers" not in out


def test_cli_run_prints_wall_time(isolated_store):
    one = CliRunner().invoke(app, ["run", "--suite", DEMO, "--target", TARGET, "--no-save"])
    many = CliRunner().invoke(
        app, ["run", "--suite", DEMO, "--target", TARGET, "--no-save", "--workers", "3"]
    )
    assert "with 1 worker" in one.output and "workers" not in one.output.split("Wall time")[1]
    assert "with 3 workers" in many.output


def test_cli_runs_shows_wall_time_and_workers(isolated_store):
    store.save_run(_timed(236.1, 4))
    store.save_run(_result([None], prefix="x"))
    res = CliRunner().invoke(app, ["runs"], env={"COLUMNS": "200"})
    rows = [table_cells(line) for line in res.output.splitlines() if " s " in line]
    assert sorted((r[7], r[8]) for r in rows) == [("236.1s", "4"), ("–", "–")]


def test_dashboard_compare_summary_has_wall_time(isolated_store):
    store.save_run(_timed(264.0, 1))
    store.save_run(_timed(236.0, 4))
    _, summary = dashboard.comparison_frame("latest-1", "latest")
    assert (summary["duration_a"], summary["workers_b"]) == (264.0, 4)
    assert {"duration_s", "workers"} <= set(dashboard.runs_frame().columns)
    assert dashboard._fmt_secs(None) == "–"
