"""The evalx command line."""

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from evalx import __version__
from evalx.judge import LLMJudge
from evalx.reports import (
    CaseRegressionNote,
    GateVerdict,
    ReportError,
    RunReport,
    parse_report_spec,
    write_report,
)
from evalx.runner import run_suite
from evalx.store import (
    StoreError,
    compare_runs,
    judged_case_ids,
    list_baselines,
    list_runs,
    load_case_spreads,
    load_run_meta,
    pin_baseline,
    regression_against_baseline,
    run_judge_usage,
    save_run,
)

app = typer.Typer(help="evalx: eval harness for LLM apps.")
console = Console()

baseline_app = typer.Typer(help="Baselines: named runs the regression gate compares against.")
app.add_typer(baseline_app, name="baseline")


@baseline_app.command("pin")
def baseline_pin(
    ref: str = typer.Argument(..., help="Run id, prefix, 'latest' or 'latest-N'."),
    name: str = typer.Option("stable", help="Baseline name to pin (re-pinning moves it)."),
    allow_errors: bool = typer.Option(
        False, help="Pin even if the run has error cases (refused by default)."
    ),
) -> None:
    """Pin a stored run as a named baseline for the regression gate."""
    try:
        baseline = pin_baseline(name, ref, allow_errors=allow_errors)
    except StoreError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(
        f"Baseline [cyan]{baseline['name']}[/cyan] pinned to run "
        f"[cyan]{baseline['run_id']}[/cyan] "
        f"(mean {baseline['mean_score']:.3f} over {baseline['n_cases']} cases)"
    )


@baseline_app.command("list")
def baseline_list() -> None:
    """List pinned baselines."""
    baselines = list_baselines()
    if not baselines:
        console.print("No baselines pinned yet. Pin one with: evalx baseline pin latest")
        return
    table = Table(title="evalx baselines")
    table.add_column("name", style="cyan")
    table.add_column("run id")
    table.add_column("mean", justify="right")
    table.add_column("cases", justify="right")
    table.add_column("pinned")
    for b in baselines:
        table.add_row(
            b["name"],
            b["run_id"],
            f"{b['mean_score']:.3f}",
            str(b["n_cases"]),
            str(b["pinned_at"]),
        )
    console.print(table)


@app.command()
def version() -> None:
    """Print the evalx version."""
    console.print(__version__)


def _ms(value: float | None) -> str:
    return "–" if value is None else f"{value:.0f}"


def _created(value) -> str:
    """Run timestamp to the minute: the table is wide enough already."""
    return value.strftime("%Y-%m-%d %H:%M") if hasattr(value, "strftime") else str(value)


def _secs(value: float | None) -> str:
    return "–" if value is None else f"{value:.1f}s"


def _usd(value: float | None) -> str:
    """Dollar amount; "–" when unknown (not judged, or the model has no price)."""
    return "–" if value is None else f"${value:.4f}"


def _pct_change(a: float | None, b: float | None) -> str:
    if a is None or b is None or a == 0:
        return ""
    return f" ({(b - a) / a:+.0%})"


def _noise_note(
    case_id: str, baseline: dict[str, float], new: dict[str, float], threshold: float
) -> str:
    """Flag a regression whose case is itself noisier than the gate threshold."""
    parts = []
    if baseline.get(case_id, 0.0) > threshold:
        parts.append(f"baseline spread {baseline[case_id]:.2f}")
    if new.get(case_id, 0.0) > threshold:
        parts.append(f"this run spread {new[case_id]:.2f}")
    if not parts:
        return ""
    return f" [yellow](noisy: {', '.join(parts)} — may be model noise)[/yellow]"


def _count_cell(value: int | None, highlight: bool = False) -> str:
    """A stored per-run count: NULL (run saved before counts existed) is unknown."""
    if value is None:
        return "–"
    if highlight and value > 0:
        return f"[bold red]{value}[/bold red]"
    return str(value)


@app.command(name="runs")
def list_runs_cmd() -> None:
    """List stored runs, newest first."""
    runs = list_runs()
    if not runs:
        console.print("No runs stored yet.")
        return
    table = Table(title="evalx runs")
    table.add_column("run id", style="cyan")
    table.add_column("suite")
    table.add_column("mean", justify="right")
    table.add_column("cases", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Skipped", justify="right")
    table.add_column("judge $", justify="right")
    table.add_column("wall", justify="right")
    table.add_column("workers", justify="right")
    table.add_column("created")
    for r in runs:
        table.add_row(
            r["run_id"],
            r["suite"],
            f"{r['mean_score']:.3f}",
            str(r["n_cases"]),
            _count_cell(r["n_errors"], highlight=True),
            _count_cell(r["n_skipped"]),
            _usd(r["judge_cost_usd"]),
            _secs(r["duration_s"]),
            _count_cell(r["workers"]),
            _created(r["created_at"]),
        )
    console.print(table)


@app.command()
def compare(
    ref_a: str = typer.Argument(..., help="Run id, prefix, 'latest' or 'latest-N'."),
    ref_b: str = typer.Argument(..., help="Run id, prefix, 'latest' or 'latest-N'."),
) -> None:
    """Compare two stored runs case by case."""
    comp = compare_runs(ref_a, ref_b)
    show_cost = comp.judge_cost_a is not None or comp.judge_cost_b is not None
    table = Table(title=f"evalx compare: {comp.run_a} -> {comp.run_b}")
    table.add_column("case", style="cyan")
    table.add_column("a", justify="right")
    table.add_column("b", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("ms a", justify="right")
    table.add_column("ms b", justify="right")
    if show_cost:
        table.add_column("judge $ a", justify="right")
        table.add_column("judge $ b", justify="right")
    for c in comp.cases:
        style = "green" if c.delta > 0 else "red" if c.delta < 0 else "dim"
        row = [
            c.case_id,
            f"{c.score_a:.2f}",
            f"{c.score_b:.2f}",
            f"{c.delta:+.2f}",
            _ms(c.latency_a),
            _ms(c.latency_b),
        ]
        if show_cost:
            row += [_usd(c.cost_a), _usd(c.cost_b)]
        table.add_row(*row, style=style)
    console.print(table)
    console.print(
        f"Mean: {comp.mean_a:.3f} -> {comp.mean_b:.3f} "
        f"([bold]{comp.delta:+.3f}[/bold]) | "
        f"{comp.improved} improved, {comp.regressed} regressed, {comp.unchanged} unchanged"
    )
    console.print(
        f"Latency (mean per case): {_ms(comp.mean_latency_a)} -> {_ms(comp.mean_latency_b)} ms"
        f"{_pct_change(comp.mean_latency_a, comp.mean_latency_b)}"
    )
    if show_cost:
        console.print(
            f"Judge cost: {_usd(comp.judge_cost_a)} -> {_usd(comp.judge_cost_b)}"
            f"{_pct_change(comp.judge_cost_a, comp.judge_cost_b)}"
        )
    console.print(
        f"Wall time (whole run): {_secs(comp.duration_a)} -> {_secs(comp.duration_b)}"
        f"{_pct_change(comp.duration_a, comp.duration_b)}"
        f" | workers {_count_cell(comp.workers_a)} -> {_count_cell(comp.workers_b)}"
    )
    if (
        comp.workers_a is not None
        and comp.workers_b is not None
        and comp.workers_a != comp.workers_b
    ):
        console.print(
            "[yellow]note: the runs used different --workers; per-case latency "
            "includes queueing at the target, so compare wall time for speed[/yellow]"
        )


@app.command()
def dashboard(
    port: int = typer.Option(8501, help="Port for the Streamlit dashboard server."),
) -> None:
    """Open the evalX run-history dashboard in your browser."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        console.print('[red]streamlit is not installed. Run: pip install -e ".[dev]"[/red]')
        raise typer.Exit(code=1)
    import subprocess
    import sys
    from pathlib import Path

    import evalx.dashboard as dashboard_mod

    url = f"http://localhost:{port}"
    console.print(f"Starting evalX dashboard at [cyan]{url}[/cyan] ...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(dashboard_mod.__file__)),
            "--server.port",
            str(port),
            "--server.headless",
            "true",
        ],
        check=False,
    )


@app.command()
def run(
    suite: str = typer.Option(..., help="Dataset directory or cases.yaml file."),
    target: str = typer.Option(..., help="App under test as 'module:attr'."),
    fail_under: float = typer.Option(0.0, help="Exit 1 if mean score is below this."),
    max_error_rate: float = typer.Option(
        0.0,
        help="Infra gate: exit 1 if the fraction of graded cases that errored "
        "(target or judge raised) exceeds this.",
    ),
    judge_model: str = typer.Option(
        None,
        help="LiteLLM judge model, e.g. 'anthropic/claude-haiku-5-5'. "
        "Skips judge scoring when omitted. Needs the provider API key in env.",
    ),
    judge_temperature: float = typer.Option(
        None,
        help="Temperature for the judge. Omitted by default: some models only "
        "accept their own default temperature.",
    ),
    workers: int = typer.Option(
        1,
        min=1,
        help="Run this many cases at once (thread pool). Results keep dataset order. "
        "Mind your target's rate limits.",
    ),
    repeats: int = typer.Option(
        1,
        min=1,
        help="Ask every case this many times; its score is the mean and the "
        "spread (max - min) shows how noisy it is.",
    ),
    no_save: bool = typer.Option(False, help="Don't save this run to the local store."),
    baseline: str = typer.Option(
        None,
        help="Baseline name: also fail if any single case drops past "
        "--fail-on-regression vs that baseline, even when the mean is fine.",
    ),
    fail_on_regression: float = typer.Option(
        0.2, help="Max allowed per-case score drop vs the baseline (needs --baseline)."
    ),
    reports: Annotated[
        list[str] | None,
        typer.Option(
            "--report",
            help="Write a report as FORMAT:PATH (json, junit or markdown). Repeatable, "
            "e.g. --report junit:evalx.xml --report markdown:$GITHUB_STEP_SUMMARY",
        ),
    ] = None,
) -> None:
    """Run a golden dataset against a target app and print the scores."""
    try:  # validate before spending a run on it
        report_specs = [parse_report_spec(spec) for spec in reports or []]
    except ReportError as exc:
        raise typer.BadParameter(str(exc), param_hint="--report") from exc
    judge = LLMJudge(judge_model, temperature=judge_temperature) if judge_model else None
    result = run_suite(suite, target, judge=judge, workers=workers, repeats=repeats)
    if not no_save:
        result.run_id = save_run(result, judge_model)
        console.print(f"Run saved: [cyan]{result.run_id}[/cyan]")

    table = Table(title=f"evalx run: {result.target} on {result.suite}")
    table.add_column("case", style="cyan")
    table.add_column("score", justify="right")
    if judge:
        table.add_column("judge", justify="right")
    if repeats > 1:
        table.add_column("spread", justify="right")
    table.add_column("latency ms", justify="right")
    for r in result.results:
        style = "green" if r.score >= 0.99 else "yellow" if r.score >= 0.5 else "red"
        row = [r.case_id, f"{r.score:.2f}"]
        if judge:
            row.append(f"{r.judge_score:.2f}" if r.judge_score is not None else "-")
        if repeats > 1:
            spread = r.spread or 0.0
            text = f"{spread:.2f}"
            row.append(f"[bold]{text}[/bold]" if spread > fail_on_regression else text)
        row.append(f"{r.latency_ms:.0f}")
        table.add_row(*row, style=style)
    console.print(table)
    for r in result.results:
        if r.error:
            console.print(f"[red]ERROR {r.case_id}: {r.error}[/red]")
    if result.skipped:
        console.print(
            f"[yellow]Skipped {len(result.skipped)} judge-only case(s) "
            f"(no --judge-model): {', '.join(result.skipped)}[/yellow]"
        )
    noisy = [r for r in result.results if (r.spread or 0.0) > fail_on_regression]
    if noisy:
        console.print(
            f"[yellow]Noisy: {len(noisy)} case(s) moved more than {fail_on_regression} "
            f"between identical attempts: "
            f"{', '.join(f'{r.case_id} ({r.spread:.2f})' for r in noisy)}[/yellow]"
        )
    attempts = f" × {repeats} attempts" if repeats > 1 else ""
    console.print(
        f"\nMean score: [bold]{result.mean_score:.3f}[/bold] over {result.n_cases} cases{attempts}"
    )
    console.print(
        f"Wall time: {_secs(result.duration_s)} with {result.workers} worker"
        f"{'' if result.workers == 1 else 's'}"
    )
    if judge:
        judge_tokens, judge_cost = run_judge_usage(result)
        if judge_tokens is not None or judge_cost is not None:
            cost = (
                _usd(judge_cost)
                if judge_cost is not None
                else f"cost unknown (no LiteLLM price for {judge_model})"
            )
            console.print(f"Judge usage: {judge_tokens or 0:,} tokens, {cost}")

    failed = False
    verdicts: list[GateVerdict] = []
    regressions: dict[str, CaseRegressionNote] = {}
    n_graded = len(result.results)
    n_errors = sum(1 for r in result.results if r.error is not None)
    error_rate = n_errors / n_graded if n_graded else 0.0
    infra_detail = f"{n_errors}/{n_graded} cases errored (allowed rate {max_error_rate:.2f})"
    if error_rate > max_error_rate:
        console.print(
            f"[red]INFRA FAIL: {n_errors}/{n_graded} cases errored "
            f"(rate {error_rate:.2f} > allowed {max_error_rate:.2f})[/red]"
        )
        failed = True
        verdicts.append(GateVerdict(gate="infra", passed=False, detail=infra_detail))
    else:
        verdicts.append(GateVerdict(gate="infra", passed=True, detail=infra_detail))

    if result.mean_score < fail_under:
        console.print(f"[red]FAIL: mean {result.mean_score:.3f} < gate {fail_under}[/red]")
        failed = True
        verdicts.append(
            GateVerdict(
                gate="mean", passed=False, detail=f"{result.mean_score:.3f} < {fail_under}"
            )
        )
    else:
        console.print("[green]Gate passed.[/green]")
        verdicts.append(
            GateVerdict(
                gate="mean", passed=True, detail=f"{result.mean_score:.3f} >= {fail_under}"
            )
        )

    if baseline is not None:
        new_scores = {r.case_id: r.score for r in result.results}
        try:
            report = regression_against_baseline(baseline, new_scores, fail_on_regression)
        except StoreError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        console.print(
            f"\nBaseline [cyan]{report.baseline_name}[/cyan] "
            f"(run {report.baseline_run_id}, threshold {report.threshold}):"
        )
        baseline_suite = load_run_meta(report.baseline_run_id)["suite"]
        if baseline_suite != result.suite:
            console.print(
                f"  [yellow]note: the baseline run came from suite {baseline_suite}, "
                f"this run is {result.suite} — only the cases they share were compared[/yellow]"
            )
        new_spreads = {r.case_id: r.spread for r in result.results if r.spread is not None}
        baseline_spreads = load_case_spreads(report.baseline_run_id) if report.regressed else {}
        for c in report.regressed:
            note = _noise_note(c.case_id, baseline_spreads, new_spreads, report.threshold)
            regressions[c.case_id] = CaseRegressionNote(
                baseline_score=c.baseline_score,
                new_score=c.new_score,
                drop=c.drop,
                noisy=bool(note),
            )
            console.print(
                f"  [red]REGRESSED {c.case_id}: "
                f"{c.baseline_score:.2f} -> {c.new_score:.2f} (drop {c.drop:.2f})[/red]"
                f"{note}"
            )
        for case_id in report.added:
            console.print(f"  [yellow]note: {case_id} is new (not in baseline) — ignored[/yellow]")
        for case_id in report.removed:
            console.print(
                f"  [yellow]note: {case_id} is in the baseline but not in this run — ignored[/yellow]"
            )
        if judge is None and report.removed:
            n_judged = len(judged_case_ids(report.baseline_run_id) & set(report.removed))
            if n_judged:
                console.print(
                    f"  [yellow]note: baseline was judged, this run is unjudged: {n_judged} "
                    "case(s) that have judge scores in the baseline were not compared[/yellow]"
                )
        detail = (
            f"baseline {report.baseline_name}: {len(report.regressed)} case(s) dropped "
            f"past {report.threshold}"
        )
        verdicts.append(GateVerdict(gate="regression", passed=report.passed, detail=detail))
        if report.passed:
            console.print("[green]No case regressed past the threshold.[/green]")
        else:
            console.print(
                f"[red]REGRESSION GATE FAILED: {len(report.regressed)} case(s) "
                f"dropped past {report.threshold}[/red]"
            )
            failed = True

    if report_specs:
        judge_tokens, judge_cost = run_judge_usage(result)
        run_report = RunReport(
            result=result,
            verdicts=verdicts,
            fail_under=fail_under,
            baseline=baseline,
            regressions=regressions,
            judge_model=judge_model,
            judge_tokens=judge_tokens,
            judge_cost_usd=judge_cost,
        )
        for fmt, path in report_specs:
            write_report(fmt, path, run_report)
            console.print(f"Report written: [cyan]{fmt}[/cyan] -> {path}")

    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
