"""Error cases must be impossible to miss.

Since the runner records per-case errors instead of aborting, an outage run
completes and is saved. These tests cover what keeps such a run visible:
per-run error counts, the --max-error-rate infra gate, the refusal to pin a
run with errors as a baseline, and the judged-vs-unjudged regression note.
"""

import duckdb
import pytest
from typer.testing import CliRunner

from evalx import dashboard, store
from evalx.cli import app
from evalx.runner import run_suite
from evalx.schemas import CaseResult, RunResult

from .helpers import table_cells

DEMO = "datasets/demo/v1"
TARGET = "examples.demo_app:answer"
CRASHING = "tests.test_runner:_crash_on_ceo"  # raises on q002 and q004


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALX_STORE", str(tmp_path / "runs.duckdb"))


def _flat(text: str) -> str:
    """Collapse whitespace: Rich wraps long lines at the terminal width."""
    return " ".join(text.split())


def _result(n_errors: int = 0, skipped: list[str] | None = None) -> RunResult:
    results = [
        CaseResult(
            case_id=f"q{i}",
            question="?",
            answer="" if i < n_errors else "a",
            check_scores=[],
            score=0.0 if i < n_errors else 1.0,
            latency_ms=1.0,
            error="target: RuntimeError: HTTP 429" if i < n_errors else None,
        )
        for i in range(3)
    ]
    return RunResult(
        suite="s",
        target="t",
        n_cases=len(results),
        mean_score=sum(r.score for r in results) / len(results),
        results=results,
        skipped=skipped or [],
    )


def _make_legacy_store() -> None:
    """A store file from before per-run counts and case errors existed."""
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


# --- 1. Per-run error counts -------------------------------------------------


def test_save_run_stores_error_and_skip_counts(isolated_store):
    run_id = store.save_run(_result(n_errors=2, skipped=["j1"]))
    meta = store.load_run_meta(run_id)
    assert meta["n_errors"] == 2
    assert meta["n_skipped"] == 1
    (listed,) = store.list_runs()
    assert (listed["n_errors"], listed["n_skipped"]) == (2, 1)


def test_clean_run_stores_zero_counts(isolated_store):
    run_id = store.save_run(_result())
    meta = store.load_run_meta(run_id)
    assert (meta["n_errors"], meta["n_skipped"]) == (0, 0)


def test_legacy_runs_have_null_counts_after_migration(isolated_store):
    _make_legacy_store()
    (old,) = store.list_runs()
    assert old["n_errors"] is None
    assert old["n_skipped"] is None
    # New runs saved into the migrated file get real counts.
    new_id = store.save_run(_result(n_errors=1))
    assert store.load_run_meta(new_id)["n_errors"] == 1


def test_cli_runs_shows_error_and_skip_columns(isolated_store):
    store.save_run(_result(n_errors=2, skipped=["j1", "j2"]))
    res = CliRunner().invoke(app, ["runs"], env={"COLUMNS": "200"})
    assert res.exit_code == 0
    assert "Errors" in res.output
    assert "Skipped" in res.output
    row = next(line for line in res.output.splitlines() if " s " in line)
    cells = table_cells(row)
    # run id, suite, mean, cases, errors, skipped, created
    assert cells[4] == "2"
    assert cells[5] == "2"


def test_cli_runs_highlights_nonzero_errors():
    from evalx.cli import _count_cell

    assert _count_cell(3, highlight=True) == "[bold red]3[/bold red]"
    assert _count_cell(0, highlight=True) == "0"
    assert _count_cell(3) == "3"
    assert _count_cell(None, highlight=True) == "–"


def test_cli_runs_renders_legacy_counts_as_dash(isolated_store):
    _make_legacy_store()
    res = CliRunner().invoke(app, ["runs"], env={"COLUMNS": "200"})
    assert res.exit_code == 0
    row = next(line for line in res.output.splitlines() if "old-run" in line)
    cells = table_cells(row)
    assert cells[4] == "–"
    assert cells[5] == "–"


def test_runs_frame_has_counts_and_dash_for_legacy(isolated_store):
    _make_legacy_store()
    new_id = store.save_run(_result(n_errors=1, skipped=["j1"]))
    df = dashboard.runs_frame()
    assert {"n_errors", "n_skipped"} <= set(df.columns)
    new = df[df["run_id"] == new_id].iloc[0]
    old = df[df["run_id"] == "old-run"].iloc[0]
    assert (new["n_errors"], new["n_skipped"]) == ("1", "1")
    assert (old["n_errors"], old["n_skipped"]) == ("–", "–")


def test_runs_frame_errors_highlighted_only_when_nonzero(isolated_store):
    _make_legacy_store()
    store.save_run(_result())
    store.save_run(_result(n_errors=2))
    df = dashboard.runs_frame()
    html = dashboard.styled_runs(df).to_html()
    assert "#ef4444" in html
    assert dashboard._errors_cell("2") != ""
    assert dashboard._errors_cell("0") == ""
    assert dashboard._errors_cell("–") == ""


# --- 2. Infra gate: --max-error-rate -----------------------------------------


def test_infra_gate_fails_on_errors_at_default_limit(isolated_store):
    res = CliRunner().invoke(app, ["run", "--suite", DEMO, "--target", CRASHING, "--no-save"])
    assert res.exit_code == 1
    out = _flat(res.output)
    assert "INFRA FAIL: 2/6 cases errored (rate 0.33 > allowed 0.00)" in out
    # Printed before the mean-gate verdict; the results table still prints.
    assert out.index("INFRA FAIL") < out.index("Gate passed.")
    assert "q002" in out
    assert "Mean score" in out


def test_infra_gate_passes_with_permissive_limit(isolated_store):
    res = CliRunner().invoke(
        app,
        ["run", "--suite", DEMO, "--target", CRASHING, "--no-save", "--max-error-rate", "1.0"],
    )
    assert res.exit_code == 0
    assert "INFRA FAIL" not in res.output
    assert "Gate passed." in res.output


def test_infra_gate_permissive_limit_still_applies_score_gate(isolated_store):
    # Error cases still count 0.0 in the mean: (1 + 0 + 0.5 + 0 + 0 + 0.5) / 6
    res = CliRunner().invoke(
        app,
        [
            "run", "--suite", DEMO, "--target", CRASHING, "--no-save",
            "--max-error-rate", "1.0", "--fail-under", "0.4",
        ],
    )  # fmt: skip
    assert res.exit_code == 1
    assert "INFRA FAIL" not in res.output
    assert "FAIL: mean 0.333 < gate 0.4" in res.output


def test_infra_gate_rate_at_limit_passes(isolated_store):
    # 2/6 = 0.333...; the gate fails only when the rate EXCEEDS the limit.
    res = CliRunner().invoke(
        app,
        ["run", "--suite", DEMO, "--target", CRASHING, "--no-save", "--max-error-rate", "0.34"],
    )
    assert res.exit_code == 0
    assert "INFRA FAIL" not in res.output


def test_infra_gate_ignores_skipped_cases(isolated_store, tmp_path):
    # One graded case that errors plus one judge-only case skipped without a
    # judge: rate is 1/1, not 1/2.
    suite = tmp_path / "cases.yaml"
    suite.write_text(
        "- id: c1\n"
        '  question: "Who is the CEO?"\n'
        "  checks:\n"
        '    - {type: contains, expected: "Jane"}\n'
        "- id: c2\n"
        '  question: "Summarise."\n'
        '  judge: {rubric: "Anything."}\n'
    )
    res = CliRunner().invoke(
        app,
        ["run", "--suite", str(suite), "--target", CRASHING, "--no-save",
         "--max-error-rate", "0.6"],
    )  # fmt: skip
    assert res.exit_code == 1
    assert "INFRA FAIL: 1/1 cases errored (rate 1.00 > allowed 0.60)" in _flat(res.output)


def test_infra_gate_clean_run_passes_by_default(isolated_store):
    res = CliRunner().invoke(app, ["run", "--suite", DEMO, "--target", TARGET, "--no-save"])
    assert res.exit_code == 0
    assert "INFRA FAIL" not in res.output


# --- 3. Refuse to pin runs that contain errors -------------------------------


def test_pin_clean_run_works(isolated_store):
    run_id = store.save_run(_result())
    assert store.pin_baseline("stable", "latest")["run_id"] == run_id


def test_pin_run_with_errors_refused_without_flag(isolated_store):
    run_id = store.save_run(_result(n_errors=2))
    with pytest.raises(store.StoreError) as exc_info:
        store.pin_baseline("stable", "latest")
    message = str(exc_info.value)
    assert f"run {run_id} has 2 error case(s)" in message
    assert "--allow-errors" in message
    assert store.get_baseline("stable") is None


def test_pin_run_with_errors_allowed_with_flag(isolated_store):
    run_id = store.save_run(_result(n_errors=2))
    assert store.pin_baseline("stable", "latest", allow_errors=True)["run_id"] == run_id


def test_pin_counts_errors_from_case_results_not_run_counts(isolated_store):
    # A run whose runs.n_errors is NULL (pre-count store) but whose
    # case_results hold an error must still be refused.
    run_id = store.save_run(_result(n_errors=1))
    con = duckdb.connect(str(store.store_path()))
    con.execute("UPDATE runs SET n_errors = NULL WHERE run_id = ?", [run_id])
    con.close()
    with pytest.raises(store.StoreError, match="1 error case"):
        store.pin_baseline("stable", "latest")


def test_cli_pin_refuses_run_with_errors(isolated_store):
    store.save_run(_result(n_errors=1))
    res = CliRunner().invoke(app, ["baseline", "pin", "latest"])
    assert res.exit_code == 1
    out = _flat(res.output)
    assert "1 error case(s)" in out
    assert "--allow-errors" in out


def test_cli_pin_allow_errors_flag(isolated_store):
    run_id = store.save_run(_result(n_errors=1))
    res = CliRunner().invoke(app, ["baseline", "pin", "latest", "--allow-errors"])
    assert res.exit_code == 0
    assert store.get_baseline("stable")["run_id"] == run_id


def test_cli_pin_clean_run(isolated_store):
    store.save_run(_result())
    res = CliRunner().invoke(app, ["baseline", "pin", "latest"])
    assert res.exit_code == 0
    assert "pinned to run" in res.output


# --- 4. Judged-vs-unjudged note in the regression report ---------------------


def _demo_baseline_with_extra_case(judge_score: float | None) -> RunResult:
    """The demo app's own scores, plus a rubric case the demo suite lacks."""
    result = run_suite(DEMO, TARGET)
    extra = CaseResult(
        case_id="q999",
        question="Summarise the year.",
        answer="a",
        check_scores=[],
        judge_score=judge_score,
        judge_reason="stub" if judge_score is not None else None,
        score=judge_score if judge_score is not None else 1.0,
        latency_ms=1.0,
    )
    results = result.results + [extra]
    return result.model_copy(update={"results": results, "n_cases": len(results)})


def test_judged_case_ids(isolated_store):
    run_id = store.save_run(_demo_baseline_with_extra_case(judge_score=0.8))
    assert store.judged_case_ids(run_id) == {"q999"}


def test_unjudged_run_against_judged_baseline_prints_note(isolated_store):
    store.save_run(_demo_baseline_with_extra_case(judge_score=0.8), judge_model="m")
    store.pin_baseline("stable", "latest")
    res = CliRunner().invoke(
        app,
        ["run", "--suite", DEMO, "--target", TARGET, "--baseline", "stable", "--no-save"],
    )
    out = _flat(res.output)
    assert "q999 is in the baseline but not in this run" in out
    assert (
        "note: baseline was judged, this run is unjudged: 1 case(s) that have judge "
        "scores in the baseline were not compared"
    ) in out
    # A note, not a failure: the shared cases match the baseline exactly.
    assert res.exit_code == 0


def test_unjudged_baseline_does_not_print_note(isolated_store):
    store.save_run(_demo_baseline_with_extra_case(judge_score=None))
    store.pin_baseline("stable", "latest")
    res = CliRunner().invoke(
        app,
        ["run", "--suite", DEMO, "--target", TARGET, "--baseline", "stable", "--no-save"],
    )
    out = _flat(res.output)
    assert "q999 is in the baseline but not in this run" in out
    assert "baseline was judged" not in out
    assert res.exit_code == 0
