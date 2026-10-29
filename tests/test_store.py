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
