"""--report: JSON, JUnit XML and Markdown reports of a run."""

import json
import xml.etree.ElementTree as ET

import pytest
from typer.testing import CliRunner

from evalx import store
from evalx.cli import app
from evalx.reports import (
    CaseRegressionNote,
    GateVerdict,
    ReportError,
    RunReport,
    parse_report_spec,
    to_json,
    to_junit,
    to_markdown,
)
from evalx.runner import run_suite
from evalx.schemas import CaseResult, RunResult

DEMO = "datasets/demo/v1"
TARGET = "examples.demo_app:answer"
CRASHING = "tests.test_runner:_crash_on_ceo"


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALX_STORE", str(tmp_path / "runs.duckdb"))


def _case(case_id, score, error=None, answer="a"):
    return CaseResult(
        case_id=case_id,
        question=f"question {case_id}?",
        answer=answer,
        check_scores=[score],
        score=score,
        latency_ms=1500.0,
        error=error,
    )


def _report(**kwargs) -> RunReport:
    results = [
        _case("ok", 1.0),
        _case("partial", 0.5),
        _case("low", 0.1),
        _case("broken", 0.0, error="target: RuntimeError: HTTP 429"),
        _case("dropped", 0.3),
    ]
    result = RunResult(
        suite="datasets/x/v1",
        target="app:answer",
        n_cases=len(results),
        mean_score=sum(r.score for r in results) / len(results),
        results=results,
        skipped=["judge-only"],
        duration_s=12.5,
        workers=2,
        repeats=1,
        run_id="20261104-000000",
    )
    defaults = {
        "result": result,
        "verdicts": [
            GateVerdict(gate="infra", passed=False, detail="1/5 cases errored"),
            GateVerdict(gate="mean", passed=True, detail="0.380 >= 0.2"),
        ],
        "fail_under": 0.2,
        "baseline": "stable",
        "regressions": {
            "dropped": CaseRegressionNote(baseline_score=1.0, new_score=0.3, drop=0.7, noisy=True)
        },
        "judge_model": "anthropic/claude-haiku-5-5",
        "judge_tokens": 1234,
        "judge_cost_usd": 0.0021,
    }
    defaults.update(kwargs)
    return RunReport(**defaults)


# --- spec parsing -------------------------------------------------------------


def test_parse_report_spec():
    fmt, path = parse_report_spec("JUnit:out/evalx.xml")
    assert fmt == "junit"
    assert str(path) == "out/evalx.xml"
    # Only the first colon splits, so Windows-style paths survive.
    assert str(parse_report_spec("json:C:/reports/run.json")[1]) == "C:/reports/run.json"


@pytest.mark.parametrize(
    ("spec", "message"),
    [("junit", "FORMAT:PATH"), ("junit:", "FORMAT:PATH"), ("html:x.html", "unknown report format")],
)
def test_bad_report_spec(spec, message):
    with pytest.raises(ReportError, match=message):
        parse_report_spec(spec)


# --- JSON ---------------------------------------------------------------------


def test_json_report():
    data = json.loads(to_json(_report()))
    assert data["passed"] is False
    assert [v["gate"] for v in data["verdicts"]] == ["infra", "mean"]
    assert data["judge"] == {
        "model": "anthropic/claude-haiku-5-5",
        "tokens": 1234,
        "cost_usd": 0.0021,
    }
    assert data["regressions"]["dropped"]["drop"] == 0.7
    assert data["run"]["run_id"] == "20261104-000000"
    assert len(data["run"]["results"]) == 5


# --- JUnit --------------------------------------------------------------------


def test_junit_report_outcomes():
    root = ET.fromstring(to_junit(_report()))
    assert root.tag == "testsuite"
    assert (root.get("tests"), root.get("failures"), root.get("errors")) == ("5", "2", "1")
    assert root.get("skipped") == "1"
    assert root.get("time") == "12.500"
    cases = {tc.get("name"): tc for tc in root.iter("testcase")}
    assert cases["ok"].find("failure") is None and cases["ok"].find("error") is None
    assert cases["partial"].find("failure") is None  # 0.5 >= --fail-under 0.2
    assert "score 0.10 < --fail-under 0.2" in cases["low"].find("failure").get("message")
    assert "HTTP 429" in cases["broken"].find("error").get("message")
    regressed = cases["dropped"].find("failure").get("message")
    assert "regressed vs baseline stable: 1.00 -> 0.30 (drop 0.70)" in regressed
    assert "noisy" in regressed
    assert cases["judge-only"].find("skipped") is not None
    assert cases["ok"].get("time") == "1.500"
    assert "answer: a" in cases["ok"].find("system-out").text
    props = {p.get("name"): p.get("value") for p in root.iter("property")}
    assert props["passed"] == "false"
    assert props["judge_cost_usd"] == "0.0021"


def test_junit_strips_characters_xml_cannot_hold():
    report = _report()
    report.result.results[0] = _case("ok", 1.0, answer="bell\x07 and nul\x00 & <tag>")
    root = ET.fromstring(to_junit(report))  # must parse
    out = next(tc for tc in root.iter("testcase") if tc.get("name") == "ok")
    text = out.find("system-out").text
    assert "\x07" not in text and "\x00" not in text
    assert "& <tag>" in text  # ordinary characters are escaped, not dropped


# --- Markdown -----------------------------------------------------------------


def test_markdown_report():
    md = to_markdown(_report())
    assert md.startswith("## ❌ evalX: `app:answer` on `datasets/x/v1`")
    assert "**Mean score 0.380** over 5 cases" in md
    assert "| infra | **FAIL** | 1/5 cases errored |" in md
    assert "| mean | pass | 0.380 >= 0.2 |" in md
    assert "wall time 12.5s (2 workers)" in md
    assert "1,234 tokens, $0.0021" in md
    assert "### Cases needing attention (4)" in md
    # Worst first; perfect cases are left out of the attention table.
    attention = md.split("### Cases needing attention")[1].split("<details>")[0]
    assert attention.index("`broken`") < attention.index("`partial`")
    assert "`ok`" not in attention
    assert "<details><summary>All cases</summary>" in md


def test_markdown_passing_run_and_table_escaping():
    report = _report(
        verdicts=[GateVerdict(gate="mean", passed=True, detail="a | b")], regressions={}
    )
    md = to_markdown(report)
    assert md.startswith("## ✅")
    assert "a \\| b" in md


# --- CLI ----------------------------------------------------------------------


def test_cli_writes_all_three_reports(isolated_store, tmp_path):
    out = tmp_path / "reports"
    res = CliRunner().invoke(
        app,
        [
            "run", "--suite", DEMO, "--target", TARGET, "--fail-under", "0.4",
            "--report", f"junit:{out / 'evalx.xml'}",
            "--report", f"markdown:{out / 'evalx.md'}",
            "--report", f"json:{out / 'evalx.json'}",
        ],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    root = ET.parse(out / "evalx.xml").getroot()
    assert root.get("tests") == "6"
    assert root.get("failures") == "2"  # q004 and q005 score 0 < 0.4
    assert (out / "evalx.md").read_text(encoding="utf-8").startswith("## ✅")
    data = json.loads((out / "evalx.json").read_text(encoding="utf-8"))
    assert data["passed"] is True
    assert data["run"]["run_id"] == store.list_runs()[0]["run_id"]


def test_cli_report_on_failing_run_still_written(isolated_store, tmp_path):
    path = tmp_path / "evalx.xml"
    res = CliRunner().invoke(
        app,
        ["run", "--suite", DEMO, "--target", CRASHING, "--no-save", "--report", f"junit:{path}"],
    )
    assert res.exit_code == 1  # infra gate
    root = ET.parse(path).getroot()
    assert root.get("errors") == "2"
    props = {p.get("name"): p.get("value") for p in root.iter("property")}
    assert props["passed"] == "false"


def test_cli_report_includes_regressions(isolated_store, tmp_path):
    perfect = run_suite(DEMO, TARGET)
    for r in perfect.results:
        r.score = 1.0
    store.save_run(perfect)
    store.pin_baseline("stable", "latest")
    path = tmp_path / "r.json"
    res = CliRunner().invoke(
        app,
        ["run", "--suite", DEMO, "--target", TARGET, "--baseline", "stable", "--no-save",
         "--report", f"json:{path}"],
    )  # fmt: skip
    assert res.exit_code == 1
    data = json.loads(path.read_text(encoding="utf-8"))
    assert {v["gate"]: v["passed"] for v in data["verdicts"]} == {
        "infra": True,
        "mean": True,
        "regression": False,
    }
    assert set(data["regressions"]) == {"q003", "q004", "q005", "q006"}


def test_cli_bad_report_spec_fails_before_running(isolated_store, tmp_path):
    res = CliRunner().invoke(
        app, ["run", "--suite", DEMO, "--target", TARGET, "--report", "html:x.html"]
    )
    assert res.exit_code == 2  # usage error
    assert "unknown report format" in res.output
    assert store.list_runs() == []  # nothing was run or saved
