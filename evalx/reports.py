"""Machine-readable run reports: JSON, JUnit XML and Markdown.

``evalx run --report FORMAT:PATH`` writes one of these after the run, so CI
can consume the results: JUnit for test-report views, Markdown for a job
summary or a PR comment, JSON for anything else. Several ``--report``
options can be given in one run.

A report holds the same verdicts the CLI prints. In JUnit, every graded
case is a ``<testcase>``: it gets an ``<error>`` when the target or judge
raised, a ``<failure>`` when it regressed past the baseline threshold or
scored below ``--fail-under``, and passes otherwise.
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from pydantic import BaseModel

from evalx.schemas import CaseResult, RunResult

FORMATS = ("json", "junit", "markdown")


class ReportError(ValueError):
    """A --report option could not be parsed."""


class GateVerdict(BaseModel):
    """One gate's outcome, as the CLI reported it."""

    gate: str  # "infra", "mean" or "regression"
    passed: bool
    detail: str


class CaseRegressionNote(BaseModel):
    baseline_score: float
    new_score: float
    drop: float
    noisy: bool = False


class RunReport(BaseModel):
    """Everything a report needs: the run plus how the gates judged it."""

    result: RunResult
    verdicts: list[GateVerdict]
    fail_under: float
    baseline: str | None = None
    regressions: dict[str, CaseRegressionNote] = {}
    judge_model: str | None = None
    judge_tokens: int | None = None
    judge_cost_usd: float | None = None

    @property
    def passed(self) -> bool:
        return all(v.passed for v in self.verdicts)


def parse_report_spec(spec: str) -> tuple[str, Path]:
    """'junit:out/evalx.xml' -> ('junit', Path('out/evalx.xml'))."""
    fmt, sep, path = spec.partition(":")
    fmt = fmt.strip().lower()
    if not sep or not path.strip():
        raise ReportError(f"--report needs FORMAT:PATH (e.g. junit:evalx.xml), got {spec!r}")
    if fmt not in FORMATS:
        raise ReportError(f"unknown report format {fmt!r}; choose from {', '.join(FORMATS)}")
    return fmt, Path(path.strip())


def render(fmt: str, report: RunReport) -> str:
    return {"json": to_json, "junit": to_junit, "markdown": to_markdown}[fmt](report)


def write_report(fmt: str, path: Path, report: RunReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(fmt, report), encoding="utf-8")


# --- JSON -------------------------------------------------------------------


def to_json(report: RunReport) -> str:
    data = {
        "passed": report.passed,
        "verdicts": [v.model_dump() for v in report.verdicts],
        "fail_under": report.fail_under,
        "baseline": report.baseline,
        "regressions": {k: v.model_dump() for k, v in report.regressions.items()},
        "judge": {
            "model": report.judge_model,
            "tokens": report.judge_tokens,
            "cost_usd": report.judge_cost_usd,
        },
        "run": report.result.model_dump(mode="json"),
    }
    return json.dumps(data, indent=2) + "\n"


# --- JUnit XML --------------------------------------------------------------

# XML 1.0 forbids most control characters, and an LLM answer can contain
# anything; ElementTree escapes <&> but would write these through as-is.
_XML_INVALID = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")


def _xml_text(text: str) -> str:
    return _XML_INVALID.sub("\ufffd", text)


def _case_outcome(case: CaseResult, report: RunReport) -> tuple[str, str] | None:
    """('error'|'failure', message) for a case that didn't pass, else None."""
    if case.error is not None:
        return "error", case.error
    reg = report.regressions.get(case.case_id)
    if reg is not None:
        noisy = " (noisy case: may be model noise)" if reg.noisy else ""
        return "failure", (
            f"regressed vs baseline {report.baseline}: {reg.baseline_score:.2f} -> "
            f"{reg.new_score:.2f} (drop {reg.drop:.2f}){noisy}"
        )
    if case.score < report.fail_under:
        return "failure", f"score {case.score:.2f} < --fail-under {report.fail_under}"
    return None


def to_junit(report: RunReport) -> str:
    result = report.result
    outcomes = {c.case_id: _case_outcome(c, report) for c in result.results}
    suite = ET.Element(
        "testsuite",
        {
            "name": f"evalx: {result.target} on {result.suite}",
            "tests": str(len(result.results)),
            "failures": str(sum(1 for o in outcomes.values() if o and o[0] == "failure")),
            "errors": str(sum(1 for o in outcomes.values() if o and o[0] == "error")),
            "skipped": str(len(result.skipped)),
            "time": f"{result.duration_s or 0.0:.3f}",
        },
    )
    props = ET.SubElement(suite, "properties")
    for name, value in (
        ("mean_score", f"{result.mean_score:.4f}"),
        ("run_id", result.run_id),
        ("workers", result.workers),
        ("repeats", result.repeats),
        ("judge_model", report.judge_model),
        ("judge_cost_usd", report.judge_cost_usd),
        ("passed", str(report.passed).lower()),
    ):
        if value is not None:
            ET.SubElement(props, "property", {"name": name, "value": str(value)})

    for case in result.results:
        el = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": result.suite,
                "name": case.case_id,
                "time": f"{case.latency_ms / 1000.0:.3f}",
            },
        )
        outcome = outcomes[case.case_id]
        if outcome is not None:
            kind, message = outcome
            message = _xml_text(message)
            ET.SubElement(el, kind, {"message": message}).text = message
        out = [f"score: {case.score:.3f}"]
        if case.judge_score is not None:
            out.append(f"judge: {case.judge_score:.3f} ({case.judge_reason or ''})")
        if case.spread is not None:
            out.append(f"spread: {case.spread:.3f} over {len(case.attempt_scores)} attempts")
        out.append(f"question: {case.question}")
        out.append(f"answer: {case.answer}")
        ET.SubElement(el, "system-out").text = _xml_text("\n".join(out))
    for case_id in result.skipped:
        el = ET.SubElement(suite, "testcase", {"classname": result.suite, "name": case_id})
        ET.SubElement(el, "skipped", {"message": "judge-only case, run without --judge-model"})

    ET.indent(suite)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(suite, "unicode") + "\n"


# --- Markdown ---------------------------------------------------------------


def _md(text: str) -> str:
    """Make text safe inside a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def to_markdown(report: RunReport) -> str:
    result = report.result
    icon = "✅" if report.passed else "❌"
    attempts = f" × {result.repeats} attempts" if (result.repeats or 1) > 1 else ""
    lines = [
        f"## {icon} evalX: `{result.target}` on `{result.suite}`",
        "",
        f"**Mean score {result.mean_score:.3f}** over {result.n_cases} cases{attempts}",
        "",
        "| Gate | Result | Detail |",
        "| --- | --- | --- |",
    ]
    for v in report.verdicts:
        lines.append(f"| {v.gate} | {'pass' if v.passed else '**FAIL**'} | {_md(v.detail)} |")

    facts = []
    if result.duration_s is not None:
        workers = result.workers or 1
        facts.append(
            f"wall time {result.duration_s:.1f}s ({workers} worker{'' if workers == 1 else 's'})"
        )
    if report.judge_model:
        cost = f"${report.judge_cost_usd:.4f}" if report.judge_cost_usd is not None else "cost unknown"
        facts.append(f"judge `{report.judge_model}`: {report.judge_tokens or 0:,} tokens, {cost}")
    if result.skipped:
        facts.append(f"{len(result.skipped)} judge-only case(s) skipped")
    if result.run_id:
        facts.append(f"run `{result.run_id}`")
    if facts:
        lines += ["", " · ".join(facts)]

    attention = [
        c for c in result.results if _case_outcome(c, report) is not None or c.score < 0.99
    ]
    if attention:
        lines += [
            "",
            f"### Cases needing attention ({len(attention)})",
            "",
            "| Case | Score | Status |",
            "| --- | ---: | --- |",
        ]
        for c in sorted(attention, key=lambda c: c.score):
            outcome = _case_outcome(c, report)
            status = f"{outcome[0]}: {outcome[1]}" if outcome else "partial"
            lines.append(f"| `{c.case_id}` | {c.score:.2f} | {_md(status)} |")

    lines += [
        "",
        "<details><summary>All cases</summary>",
        "",
        "| Case | Score | Judge | Latency ms |",
        "| --- | ---: | ---: | ---: |",
    ]
    for c in result.results:
        judge = f"{c.judge_score:.2f}" if c.judge_score is not None else "–"
        lines.append(f"| `{c.case_id}` | {c.score:.2f} | {judge} | {c.latency_ms:.0f} |")
    lines += ["", "</details>", ""]
    return "\n".join(lines)
