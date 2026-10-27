"""DuckDB-backed run store: every run is saved, so runs can be compared.

Location defaults to ``.evalx/runs.duckdb`` in the working directory and can
be overridden with the ``EVALX_STORE`` environment variable.
"""

import os
from datetime import datetime
from pathlib import Path

import duckdb
from pydantic import BaseModel

from evalx.schemas import RunResult


class StoreError(Exception):
    """Raised when a run reference cannot be resolved."""


def store_path() -> Path:
    return Path(os.environ.get("EVALX_STORE", ".evalx/runs.duckdb"))


_SCHEMA_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    suite VARCHAR,
    target VARCHAR,
    judge_model VARCHAR,
    mean_score DOUBLE,
    n_cases INTEGER,
    created_at TIMESTAMP DEFAULT now()
)
"""

_SCHEMA_CASES = """
CREATE TABLE IF NOT EXISTS case_results (
    run_id VARCHAR,
    case_id VARCHAR,
    score DOUBLE,
    judge_score DOUBLE,
    latency_ms DOUBLE
)
"""


def _connect(path: Path | None = None):
    path = path or store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(_SCHEMA_RUNS)
    con.execute(_SCHEMA_CASES)
    return con


def _new_run_id(con) -> str:
    base = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id, n = base, 1
    while con.execute("SELECT 1 FROM runs WHERE run_id = ?", [run_id]).fetchone():
        n += 1
        run_id = f"{base}-{n}"
    return run_id


def save_run(result: RunResult, judge_model: str | None = None, path: Path | None = None) -> str:
    """Persist a run and its per-case scores. Returns the run id."""
    con = _connect(path)
    try:
        run_id = _new_run_id(con)
        con.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, now())",
            [run_id, result.suite, result.target, judge_model, result.mean_score, result.n_cases],
        )
        con.executemany(
            "INSERT INTO case_results VALUES (?, ?, ?, ?, ?)",
            [(run_id, r.case_id, r.score, r.judge_score, r.latency_ms) for r in result.results],
        )
    finally:
        con.close()
    return run_id


def list_runs(path: Path | None = None) -> list[dict]:
    """All stored runs, newest first."""
    con = _connect(path)
    try:
        rows = con.execute(
            "SELECT run_id, suite, target, judge_model, mean_score, n_cases, created_at"
            " FROM runs ORDER BY created_at DESC"
        ).fetchall()
    finally:
        con.close()
    cols = ["run_id", "suite", "target", "judge_model", "mean_score", "n_cases", "created_at"]
    return [dict(zip(cols, row)) for row in rows]


def resolve_run_id(ref: str, path: Path | None = None) -> str:
    """Resolve 'latest', 'latest-N', a unique id prefix, or a full id."""
    runs = list_runs(path)
    if not runs:
        raise StoreError("no runs stored yet")
    if ref == "latest":
        return runs[0]["run_id"]
    if ref.startswith("latest-"):
        # "latest-N" = the run N positions before the newest one.
        try:
            n = int(ref.split("-", 1)[1])
        except ValueError:
            raise StoreError(f"invalid run reference: {ref!r}") from None
        if n < 1 or n >= len(runs):
            raise StoreError(f"only {len(runs)} run(s) stored, asked for {ref!r}")
        return runs[n]["run_id"]
    matches = [r["run_id"] for r in runs if r["run_id"].startswith(ref)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise StoreError(f"no run matches {ref!r}")
    raise StoreError(f"ambiguous {ref!r}: matches {matches}")


def load_run_meta(run_id: str, path: Path | None = None) -> dict:
    con = _connect(path)
    try:
        row = con.execute(
            "SELECT run_id, suite, target, judge_model, mean_score, n_cases, created_at"
            " FROM runs WHERE run_id = ?",
            [run_id],
        ).fetchone()
    finally:
        con.close()
    if row is None:
        raise StoreError(f"unknown run id: {run_id!r}")
    cols = ["run_id", "suite", "target", "judge_model", "mean_score", "n_cases", "created_at"]
    return dict(zip(cols, row))


def load_case_scores(run_id: str, path: Path | None = None) -> dict[str, float]:
    con = _connect(path)
    try:
        rows = con.execute(
            "SELECT case_id, score FROM case_results WHERE run_id = ?", [run_id]
        ).fetchall()
    finally:
        con.close()
    return {case_id: score for case_id, score in rows}


def load_case_details(run_id: str, path: Path | None = None) -> list[dict]:
    """Per-case score, judge score and latency for one run, in stored order."""
    con = _connect(path)
    try:
        rows = con.execute(
            "SELECT case_id, score, judge_score, latency_ms FROM case_results WHERE run_id = ?",
            [run_id],
        ).fetchall()
    finally:
        con.close()
    cols = ["case_id", "score", "judge_score", "latency_ms"]
    return [dict(zip(cols, row)) for row in rows]


class CaseDelta(BaseModel):
    case_id: str
    score_a: float
    score_b: float
    delta: float


class RunComparison(BaseModel):
    run_a: str
    run_b: str
    mean_a: float
    mean_b: float
    delta: float
    improved: int
    regressed: int
    unchanged: int
    cases: list[CaseDelta]


def compare_runs(ref_a: str, ref_b: str, path: Path | None = None) -> RunComparison:
    """Compare two stored runs case by case (inner join on case id)."""
    run_a = resolve_run_id(ref_a, path)
    run_b = resolve_run_id(ref_b, path)
    meta_a = load_run_meta(run_a, path)
    meta_b = load_run_meta(run_b, path)
    scores_a = load_case_scores(run_a, path)
    scores_b = load_case_scores(run_b, path)

    cases = [
        CaseDelta(
            case_id=c,
            score_a=scores_a[c],
            score_b=scores_b[c],
            delta=scores_b[c] - scores_a[c],
        )
        for c in sorted(set(scores_a) & set(scores_b))
    ]
    improved = sum(1 for c in cases if c.delta > 1e-9)
    regressed = sum(1 for c in cases if c.delta < -1e-9)
    return RunComparison(
        run_a=run_a,
        run_b=run_b,
        mean_a=meta_a["mean_score"],
        mean_b=meta_b["mean_score"],
        delta=meta_b["mean_score"] - meta_a["mean_score"],
        improved=improved,
        regressed=regressed,
        unchanged=len(cases) - improved - regressed,
        cases=cases,
    )
