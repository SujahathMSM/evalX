"""The tiny RAG example behaves exactly as its README walkthrough says."""

import pytest
from typer.testing import CliRunner

from evalx import store
from evalx.cli import app
from evalx.runner import run_suite

SUITE = "examples/tiny_rag/cases.yaml"
TARGET = "examples.tiny_rag.app:answer"


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALX_STORE", str(tmp_path / "runs.duckdb"))


def _scores(monkeypatch, model: str) -> dict[str, float]:
    monkeypatch.setenv("TINY_RAG_MODEL", model)
    result = run_suite(SUITE, TARGET)
    return {r.case_id: r.score for r in result.results}


def test_v1_half_passes_torque_on_the_wrong_unit(monkeypatch):
    scores = _scores(monkeypatch, "v1")
    assert scores == {
        "torque-value": 0.5,
        "tire-gas": 1.0,
        "cable-rule": 1.0,
        "refuse-offtopic": 1.0,
    }


def test_v2_improves_torque(monkeypatch):
    assert set(_scores(monkeypatch, "v2").values()) == {1.0}


def test_v3_stops_refusing(monkeypatch):
    scores = _scores(monkeypatch, "v3")
    assert scores["refuse-offtopic"] == 0.0
    assert sum(scores.values()) / len(scores) == 0.75


def test_readme_walkthrough(isolated_store, monkeypatch):
    runner = CliRunner()
    monkeypatch.delenv("TINY_RAG_MODEL", raising=False)
    assert runner.invoke(app, ["run", "--suite", SUITE, "--target", TARGET]).exit_code == 0
    assert runner.invoke(app, ["baseline", "pin", "latest", "--name", "tiny-rag"]).exit_code == 0

    monkeypatch.setenv("TINY_RAG_MODEL", "v2")
    v2 = runner.invoke(app, ["run", "--suite", SUITE, "--target", TARGET, "--baseline", "tiny-rag"])
    assert v2.exit_code == 0
    assert "No case regressed" in v2.output
    comp = store.compare_runs("latest-1", "latest")
    assert (comp.improved, comp.regressed, comp.unchanged) == (1, 0, 3)

    monkeypatch.setenv("TINY_RAG_MODEL", "v3")
    v3 = runner.invoke(
        app,
        ["run", "--suite", SUITE, "--target", TARGET, "--baseline", "tiny-rag"],
        env={"COLUMNS": "200"},
    )
    assert v3.exit_code == 1
    assert "REGRESSED refuse-offtopic: 1.00 -> 0.00 (drop 1.00)" in v3.output
