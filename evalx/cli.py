"""The evalx command line."""

import typer
from rich.console import Console
from rich.table import Table

from evalx import __version__
from evalx.judge import LLMJudge
from evalx.runner import run_suite

app = typer.Typer(help="evalx: eval harness for LLM apps.")
console = Console()


@app.command()
def version() -> None:
    """Print the evalx version."""
    console.print(__version__)


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
) -> None:
    """Run a golden dataset against a target app and print the scores."""
    judge = LLMJudge(judge_model) if judge_model else None
    result = run_suite(suite, target, judge=judge)

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
