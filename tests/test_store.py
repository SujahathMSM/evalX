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
