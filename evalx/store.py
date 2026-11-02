"""DuckDB-backed run store: every run is saved, so runs can be compared.

Location defaults to ``.evalx/runs.duckdb`` in the working directory and can
be overridden with the ``EVALX_STORE`` environment variable.
"""

import json
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
    latency_ms DOUBLE,
    question VARCHAR,
    answer VARCHAR,
    judge_reason VARCHAR,
    checks_json VARCHAR
)
"""

# Detail columns added after the first store release. Fresh databases get
# them from the schema above; older files are migrated in _connect. Rows
# saved before the migration hold NULL here, which readers surface as None.
_CASE_DETAIL_COLUMNS = ("question", "answer", "judge_reason", "checks_json")


_SCHEMA_BASELINES = """
CREATE TABLE IF NOT EXISTS baselines (
    name VARCHAR PRIMARY KEY,
    run_id VARCHAR,
    pinned_at TIMESTAMP DEFAULT now()
)
"""


def _connect(path: Path | None = None):
    path = path or store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(_SCHEMA_RUNS)
    con.execute(_SCHEMA_CASES)
    con.execute(_SCHEMA_BASELINES)
    for col in _CASE_DETAIL_COLUMNS:
        con.execute(f"ALTER TABLE case_results ADD COLUMN IF NOT EXISTS {col} VARCHAR")
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
            "INSERT INTO case_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    run_id,
                    r.case_id,
                    r.score,
                    r.judge_score,
                    r.latency_ms,
                    r.question,
                    r.answer,
                    r.judge_reason,
                    json.dumps([c.model_dump() for c in r.checks]),
                )
                for r in result.results
            ],
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
    """Per-case detail for one run, in stored order.

    ``question``, ``answer``, ``judge_reason`` and ``checks`` are None for
    runs saved before the store kept case detail (see _CASE_DETAIL_COLUMNS).
    """
    con = _connect(path)
    try:
        rows = con.execute(
            "SELECT case_id, score, judge_score, latency_ms, question, answer,"
            " judge_reason, checks_json FROM case_results WHERE run_id = ?",
            [run_id],
        ).fetchall()
    finally:
        con.close()
    cols = ["case_id", "score", "judge_score", "latency_ms", "question", "answer", "judge_reason"]
    details = []
    for row in rows:
        detail = dict(zip(cols, row[:7]))
        detail["checks"] = json.loads(row[7]) if row[7] else None
        details.append(detail)
    return details


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


# --- Baselines and the per-case regression gate (M5) -----------------------
#
# A baseline is a named pointer at a stored run: "the last run we trusted".
# The regression gate then asks a stricter question than the mean gate --
# not "is the average still fine?" but "did ANY single case fall past the
# threshold?", because a mean can improve while one case craters.


def pin_baseline(name: str, ref: str, path: Path | None = None) -> dict:
    """Point baseline ``name`` at the run ``ref`` resolves to.

    Re-pinning an existing name moves it to the new run. Returns the
    baseline joined with the run's meta (mean, n_cases).
    """
    run_id = resolve_run_id(ref, path)
    meta = load_run_meta(run_id, path)
    con = _connect(path)
    try:
        con.execute(
            "INSERT OR REPLACE INTO baselines VALUES (?, ?, now())",
            [name, run_id],
        )
        row = con.execute(
            "SELECT pinned_at FROM baselines WHERE name = ?", [name]
        ).fetchone()
    finally:
        con.close()
    return {
        "name": name,
        "run_id": run_id,
        "mean_score": meta["mean_score"],
        "n_cases": meta["n_cases"],
        "pinned_at": row[0],
    }


def get_baseline(name: str, path: Path | None = None) -> dict | None:
    """One baseline by name (joined with its run's meta), or None."""
    con = _connect(path)
    try:
        row = con.execute(
            "SELECT b.name, b.run_id, b.pinned_at, r.mean_score, r.n_cases"
            " FROM baselines b JOIN runs r ON r.run_id = b.run_id"
            " WHERE b.name = ?",
            [name],
        ).fetchone()
    finally:
        con.close()
    if row is None:
        return None
    cols = ["name", "run_id", "pinned_at", "mean_score", "n_cases"]
    return dict(zip(cols, row))


def list_baselines(path: Path | None = None) -> list[dict]:
    """All pinned baselines, most recently pinned first."""
    con = _connect(path)
    try:
        rows = con.execute(
            "SELECT b.name, b.run_id, b.pinned_at, r.mean_score, r.n_cases"
            " FROM baselines b JOIN runs r ON r.run_id = b.run_id"
            " ORDER BY b.pinned_at DESC"
        ).fetchall()
    finally:
        con.close()
    cols = ["name", "run_id", "pinned_at", "mean_score", "n_cases"]
    return [dict(zip(cols, row)) for row in rows]


class CaseRegression(BaseModel):
    case_id: str
    baseline_score: float
    new_score: float
    drop: float


class RegressionReport(BaseModel):
    baseline_name: str
    baseline_run_id: str
    threshold: float
    regressed: list[CaseRegression]
    added: list[str]
    removed: list[str]
    passed: bool


def regression_against_baseline(
    name: str,
    new_scores: dict[str, float],
    threshold: float,
    path: Path | None = None,
) -> RegressionReport:
    """Compare fresh per-case scores against a pinned baseline.

    A case regresses when its score dropped by MORE than ``threshold``
    (a drop of exactly the threshold still passes). Cases that exist on
    only one side are reported as added/removed and never fail the gate:
    a changed dataset is a warning, not a regression. But if the two runs
    share NO cases at all, the comparison itself is meaningless (almost
    certainly the wrong baseline), so StoreError is raised instead of
    reporting a pass.
    """
    baseline = get_baseline(name, path)
    if baseline is None:
        raise StoreError(
            f"no baseline named {name!r}; pin one first: evalx baseline pin latest --name {name}"
        )
    baseline_scores = load_case_scores(baseline["run_id"], path)
    shared = set(baseline_scores) & set(new_scores)
    if not shared:
        # Comparing runs from different datasets: every case would land in
        # added/removed and the gate would "pass" a comparison that never
        # happened. Refuse loudly instead of reporting a false pass.
        raise StoreError(
            f"baseline {name!r} (run {baseline['run_id']}) and this run share no cases: "
            "a baseline only makes sense against the same dataset "
            f"(baseline has {len(baseline_scores)} case(s), this run has {len(new_scores)})"
        )
    regressed = [
        CaseRegression(
            case_id=c,
            baseline_score=baseline_scores[c],
            new_score=new_scores[c],
            drop=baseline_scores[c] - new_scores[c],
        )
        for c in sorted(shared)
        if baseline_scores[c] - new_scores[c] > threshold + 1e-9
    ]
    return RegressionReport(
        baseline_name=name,
        baseline_run_id=baseline["run_id"],
        threshold=threshold,
        regressed=regressed,
        added=sorted(set(new_scores) - set(baseline_scores)),
        removed=sorted(set(baseline_scores) - set(new_scores)),
        passed=not regressed,
    )
