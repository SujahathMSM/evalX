"""The evalx command line."""

import typer
from rich.console import Console
from rich.table import Table

from evalx import __version__
from evalx.judge import LLMJudge
from evalx.runner import run_suite
from evalx.store import compare_runs, list_runs, save_run

app = typer.Typer(help="evalx: eval harness for LLM apps.")
console = Console()


@app.command()
def version() -> None:
    """Print the evalx version."""
    console.print(__version__)


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
    table.add_column("created")
    for r in runs:
        table.add_row(
            r["run_id"],
            r["suite"],
            f"{r['mean_score']:.3f}",
            str(r["n_cases"]),
            str(r["created_at"]),
        )
    console.print(table)


@app.command()
def compare(
    ref_a: str = typer.Argument(..., help="Run id, prefix, 'latest' or 'latest-N'."),
    ref_b: str = typer.Argument(..., help="Run id, prefix, 'latest' or 'latest-N'."),
) -> None:
    """Compare two stored runs case by case."""
    comp = compare_runs(ref_a, ref_b)
    table = Table(title=f"evalx compare: {comp.run_a} -> {comp.run_b}")
    table.add_column("case", style="cyan")
    table.add_column("a", justify="right")
    table.add_column("b", justify="right")
    table.add_column("delta", justify="right")
    for c in comp.cases:
        style = "green" if c.delta > 0 else "red" if c.delta < 0 else "dim"
        table.add_row(
            c.case_id, f"{c.score_a:.2f}", f"{c.score_b:.2f}", f"{c.delta:+.2f}", style=style
        )
    console.print(table)
    console.print(
        f"Mean: {comp.mean_a:.3f} -> {comp.mean_b:.3f} "
        f"([bold]{comp.delta:+.3f}[/bold]) | "
        f"{comp.improved} improved, {comp.regressed} regressed, {comp.unchanged} unchanged"
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
    no_save: bool = typer.Option(False, help="Don't save this run to the local store."),
) -> None:
    """Run a golden dataset against a target app and print the scores."""
    judge = LLMJudge(judge_model, temperature=judge_temperature) if judge_model else None
    result = run_suite(suite, target, judge=judge)
    if not no_save:
        result.run_id = save_run(result, judge_model)
        console.print(f"Run saved: [cyan]{result.run_id}[/cyan]")

    table = Table(title=f"evalx run: {result.target} on {result.suite}")
    table.add_column("case", style="cyan")
    table.add_column("score", justify="right")
    if judge:
        table.add_column("judge", justify="right")
    table.add_column("latency ms", justify="right")
    for r in result.results:
        style = "green" if r.score >= 0.99 else "yellow" if r.score >= 0.5 else "red"
        row = [r.case_id, f"{r.score:.2f}"]
        if judge:
            row.append(f"{r.judge_score:.2f}" if r.judge_score is not None else "-")
        row.append(f"{r.latency_ms:.0f}")
        table.add_row(*row, style=style)
    console.print(table)
    console.print(f"\nMean score: [bold]{result.mean_score:.3f}[/bold] over {result.n_cases} cases")

    if result.mean_score < fail_under:
        console.print(f"[red]FAIL: mean {result.mean_score:.3f} < gate {fail_under}[/red]")
        raise typer.Exit(code=1)
    console.print("[green]Gate passed.[/green]")


if __name__ == "__main__":
    app()
