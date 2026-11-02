"""Tests for the DuckDB run store and run comparison."""

import pytest
from typer.testing import CliRunner

from evalx import store
from evalx.cli import app
from evalx.runner import run_suite
from evalx.schemas import CaseResult, RunResult

DEMO = "datasets/demo/v1"
TARGET = "examples.demo_app:answer"


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALX_STORE", str(tmp_path / "runs.duckdb"))


def _fake_result(scores: list[float]) -> RunResult:
    return RunResult(
        suite="s",
        target="t",
        n_cases=len(scores),
        mean_score=sum(scores) / len(scores),
        results=[
            CaseResult(
                case_id=f"q{i}",
                question="?",
                answer="a",
                check_scores=[s],
                score=s,
                latency_ms=1.0,
            )
            for i, s in enumerate(scores)
        ],
    )


def test_save_and_list_roundtrip(isolated_store):
    result = run_suite(DEMO, TARGET)
    run_id = store.save_run(result, judge_model="anthropic/claude-haiku-5-5")
    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["mean_score"] == 0.5
    assert runs[0]["judge_model"] == "anthropic/claude-haiku-5-5"
    assert store.load_case_scores(run_id)["q001"] == 1.0


def test_run_ids_unique(isolated_store):
    assert store.save_run(_fake_result([1.0])) != store.save_run(_fake_result([1.0]))


def test_resolve_latest_and_prefix(isolated_store):
    first = store.save_run(_fake_result([0.5]))
    second = store.save_run(_fake_result([1.0]))
    assert store.resolve_run_id("latest") == second
    assert store.resolve_run_id("latest-1") == first
    assert store.resolve_run_id(second) == second
    # the shared date prefix is genuinely ambiguous
    with pytest.raises(store.StoreError):
        store.resolve_run_id(second[:8])
    # ...but a longer prefix unique to the second run resolves
    assert store.resolve_run_id(second[:16]) == second


def test_resolve_errors(isolated_store):
    with pytest.raises(store.StoreError):
        store.resolve_run_id("latest")  # empty store
    store.save_run(_fake_result([1.0]))
    with pytest.raises(store.StoreError):
        store.resolve_run_id("nope")
    with pytest.raises(store.StoreError):
        store.resolve_run_id("latest-5")
    with pytest.raises(store.StoreError):
        store.resolve_run_id("latest-x")


def test_compare_runs(isolated_store):
    store.save_run(_fake_result([1.0, 0.5]))
    store.save_run(_fake_result([1.0, 1.0]))
    comp = store.compare_runs("latest-1", "latest")
    assert comp.mean_a == 0.75
    assert comp.mean_b == 1.0
    assert comp.delta == pytest.approx(0.25)
    assert comp.improved == 1
    assert comp.regressed == 0
    assert comp.unchanged == 1
    by_id = {c.case_id: c for c in comp.cases}
    assert by_id["q1"].delta == pytest.approx(0.5)


def test_compare_regression(isolated_store):
    store.save_run(_fake_result([1.0, 1.0]))
    store.save_run(_fake_result([0.0, 1.0]))
    comp = store.compare_runs("latest-1", "latest")
    assert comp.delta == pytest.approx(-0.5)
    assert comp.regressed == 1


def test_cli_compare(isolated_store):
    store.save_run(_fake_result([1.0, 0.5]))
    store.save_run(_fake_result([1.0, 1.0]))
    res = CliRunner().invoke(app, ["compare", "latest-1", "latest"])
    assert res.exit_code == 0
    assert "+0.250" in res.output
    assert "1 improved" in res.output


def test_cli_runs_empty(isolated_store):
    res = CliRunner().invoke(app, ["runs"])
    assert res.exit_code == 0
    assert "No runs stored yet" in res.output


def test_store_path_env_override(tmp_path, monkeypatch):
    custom = tmp_path / "custom.duckdb"
    monkeypatch.setenv("EVALX_STORE", str(custom))
    store.save_run(_fake_result([1.0]))
    assert custom.exists()


def test_case_details_full_roundtrip(isolated_store):
    result = run_suite(DEMO, TARGET)
    run_id = store.save_run(result)
    details = {d["case_id"]: d for d in store.load_case_details(run_id)}
    q003 = details["q003"]
    assert q003["question"] == "Why did operating costs rise in Q3?"
    assert "supply chain" in q003["answer"]
    assert [c["expected"] for c in q003["checks"]] == ["warehouse", "supply chain"]
    assert [c["score"] for c in q003["checks"]] == [0.0, 1.0]
    assert details["q001"]["judge_reason"] is None


def test_case_details_judge_reason_roundtrip(isolated_store):
    result = RunResult(
        suite="s",
        target="t",
        n_cases=1,
        mean_score=0.5,
        results=[
            CaseResult(
                case_id="q1",
                question="?",
                answer="a",
                check_scores=[],
                judge_score=0.5,
                judge_reason="partial: named one cause",
                score=0.5,
                latency_ms=3.0,
            )
        ],
    )
    run_id = store.save_run(result, judge_model="m")
    (detail,) = store.load_case_details(run_id)
    assert detail["judge_reason"] == "partial: named one cause"
    assert detail["judge_score"] == 0.5


def test_store_migrates_old_schema(isolated_store):
    import duckdb

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
    con.execute("INSERT INTO case_results VALUES ('old-run', 'q001', 1.0, NULL, 1.0)")
    con.close()

    # Opening the old file through the store migrates it in place.
    assert store.list_runs()[0]["run_id"] == "old-run"
    (detail,) = store.load_case_details("old-run")
    assert detail["score"] == 1.0
    assert detail["question"] is None
    assert detail["checks"] is None

    new_id = store.save_run(_fake_result([1.0]))
    (new_detail,) = store.load_case_details(new_id)
    assert new_detail["question"] == "?"


# --- M5: baselines and the per-case regression gate -------------------------


def test_pin_and_get_baseline(isolated_store):
    run_id = store.save_run(_fake_result([1.0, 0.5]))
    pinned = store.pin_baseline("stable", "latest")
    assert pinned["run_id"] == run_id
    assert pinned["mean_score"] == pytest.approx(0.75)
    assert pinned["n_cases"] == 2
    got = store.get_baseline("stable")
    assert got["run_id"] == run_id
    assert [b["name"] for b in store.list_baselines()] == ["stable"]


def test_repin_baseline_moves_it(isolated_store):
    store.save_run(_fake_result([1.0]))
    second = store.save_run(_fake_result([0.5]))
    store.pin_baseline("stable", "latest-1")
    store.pin_baseline("stable", "latest")
    assert store.get_baseline("stable")["run_id"] == second
    assert len(store.list_baselines()) == 1


def test_pin_unknown_run_raises(isolated_store):
    store.save_run(_fake_result([1.0]))
    with pytest.raises(store.StoreError, match="no run matches"):
        store.pin_baseline("stable", "19990101")


def test_baseline_missing(isolated_store):
    assert store.get_baseline("nope") is None
    assert store.list_baselines() == []
    res = CliRunner().invoke(app, ["baseline", "list"])
    assert res.exit_code == 0
    assert "No baselines pinned yet" in res.output


def test_regression_flags_drop_past_threshold(isolated_store):
    store.save_run(_fake_result([1.0, 1.0, 0.5]))
    store.pin_baseline("stable", "latest")
    report = store.regression_against_baseline(
        "stable", {"q0": 1.0, "q1": 0.6, "q2": 0.5}, threshold=0.2
    )
    assert not report.passed
    assert [c.case_id for c in report.regressed] == ["q1"]
    assert report.regressed[0].drop == pytest.approx(0.4)
    assert report.added == [] and report.removed == []


def test_regression_drop_exactly_at_threshold_passes(isolated_store):
    store.save_run(_fake_result([1.0]))
    store.pin_baseline("stable", "latest")
    report = store.regression_against_baseline("stable", {"q0": 0.8}, threshold=0.2)
    assert report.passed
    assert report.regressed == []


def test_regression_added_removed_warn_only(isolated_store):
    store.save_run(_fake_result([1.0, 1.0]))
    store.pin_baseline("stable", "latest")
    report = store.regression_against_baseline(
        "stable", {"q0": 1.0, "q9": 0.0}, threshold=0.2
    )
    assert report.passed
    assert report.added == ["q9"]
    assert report.removed == ["q1"]


def test_regression_missing_baseline_raises(isolated_store):
    with pytest.raises(store.StoreError, match="no baseline named"):
        store.regression_against_baseline("ghost", {"q0": 1.0}, threshold=0.2)


def test_cli_baseline_pin_and_list(isolated_store):
    result = run_suite(DEMO, TARGET)
    store.save_run(result)
    res = CliRunner().invoke(app, ["baseline", "pin", "latest"])
    assert res.exit_code == 0
    assert "stable" in res.output and "0.500" in res.output
    res = CliRunner().invoke(app, ["baseline", "list"])
    assert res.exit_code == 0
    assert "stable" in res.output


def _perfect_demo_result() -> RunResult:
    ids = ["q001", "q002", "q003", "q004", "q005", "q006"]
    return RunResult(
        suite=DEMO,
        target=TARGET,
        n_cases=len(ids),
        mean_score=1.0,
        results=[
            CaseResult(
                case_id=i,
                question="?",
                answer="a",
                check_scores=[1.0],
                score=1.0,
                latency_ms=1.0,
            )
            for i in ids
        ],
    )


def test_cli_run_regression_gate_fails(isolated_store):
    store.save_run(_perfect_demo_result())
    store.pin_baseline("stable", "latest")
    res = CliRunner().invoke(
        app,
        ["run", "--suite", DEMO, "--target", TARGET, "--baseline", "stable", "--no-save"],
    )
    assert res.exit_code == 1
    assert "REGRESSED q003" in res.output
    assert "REGRESSION GATE FAILED" in res.output


def test_cli_run_regression_gate_passes_against_itself(isolated_store):
    result = run_suite(DEMO, TARGET)
    store.save_run(result)
    store.pin_baseline("stable", "latest")
    res = CliRunner().invoke(
        app,
        ["run", "--suite", DEMO, "--target", TARGET, "--baseline", "stable", "--no-save"],
    )
    assert res.exit_code == 0
    assert "No case regressed" in res.output
