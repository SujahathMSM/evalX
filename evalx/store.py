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
    created_at TIMESTAMP DEFAULT now(),
    n_errors INTEGER,
    n_skipped INTEGER,
    judge_tokens INTEGER,
    judge_cost_usd DOUBLE
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
    checks_json VARCHAR,
    error VARCHAR,
    judge_prompt_tokens INTEGER,
    judge_completion_tokens INTEGER,
    judge_cost_usd DOUBLE
)
"""

# Detail columns added after the first store release. Fresh databases get
# them from the schema above; older files are migrated in _connect. Rows
# saved before the migration hold NULL here, which readers surface as None.
_CASE_DETAIL_COLUMNS = ("question", "answer", "judge_reason", "checks_json", "error")

# Judge usage per case, added later still (NULL: not judged, or not reported).
_CASE_USAGE_COLUMNS = (
    ("judge_prompt_tokens", "INTEGER"),
    ("judge_completion_tokens", "INTEGER"),
    ("judge_cost_usd", "DOUBLE"),
)

# Per-run columns added after the first store release, migrated the same way.
# Runs saved before the migration hold NULL: unknown, not zero.
_RUN_EXTRA_COLUMNS = (
    ("n_errors", "INTEGER"),
    ("n_skipped", "INTEGER"),
    ("judge_tokens", "INTEGER"),
    ("judge_cost_usd", "DOUBLE"),
)


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
    for col, sql_type in _CASE_USAGE_COLUMNS:
        con.execute(f"ALTER TABLE case_results ADD COLUMN IF NOT EXISTS {col} {sql_type}")
    for col, sql_type in _RUN_EXTRA_COLUMNS:
        con.execute(f"ALTER TABLE runs ADD COLUMN IF NOT EXISTS {col} {sql_type}")
    return con


def _new_run_id(con) -> str:
    base = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_id, n = base, 1
    while con.execute("SELECT 1 FROM runs WHERE run_id = ?", [run_id]).fetchone():
        n += 1
        run_id = f"{base}-{n}"
    return run_id


def run_judge_usage(result: RunResult) -> tuple[int | None, float | None]:
    """Total judge tokens and cost for a run.

    Each total is None when no case reported it (no judge, or a model
    LiteLLM can't price), so "unknown" never reads as a free run.
    """
    tokens = [
        (r.judge_prompt_tokens or 0) + (r.judge_completion_tokens or 0)
        for r in result.results
        if r.judge_prompt_tokens is not None or r.judge_completion_tokens is not None
    ]
    costs = [r.judge_cost_usd for r in result.results if r.judge_cost_usd is not None]
    return (sum(tokens) if tokens else None, sum(costs) if costs else None)


def save_run(result: RunResult, judge_model: str | None = None, path: Path | None = None) -> str:
    """Persist a run and its per-case scores. Returns the run id."""
    con = _connect(path)
    try:
        run_id = _new_run_id(con)
        n_errors = sum(1 for r in result.results if r.error is not None)
        judge_tokens, judge_cost = run_judge_usage(result)
        con.execute(
            "INSERT INTO runs (run_id, suite, target, judge_model, mean_score, n_cases,"
            " created_at, n_errors, n_skipped, judge_tokens, judge_cost_usd)"
            " VALUES (?, ?, ?, ?, ?, ?, now(), ?, ?, ?, ?)",
            [
                run_id,
                result.suite,
                result.target,
                judge_model,
                result.mean_score,
                result.n_cases,
                n_errors,
                len(result.skipped),
                judge_tokens,
                judge_cost,
            ],
        )
        con.executemany(
            "INSERT INTO case_results (run_id, case_id, score, judge_score, latency_ms, question,"
            " answer, judge_reason, checks_json, error, judge_prompt_tokens,"
            " judge_completion_tokens, judge_cost_usd)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    r.error,
                    r.judge_prompt_tokens,
                    r.judge_completion_tokens,
                    r.judge_cost_usd,
                )
                for r in result.results
            ],
        )
    finally:
        con.close()
    return run_id


_RUN_META_COLUMNS = [
    "run_id",
    "suite",
    "target",
    "judge_model",
    "mean_score",
    "n_cases",
    "created_at",
    "n_errors",
    "n_skipped",
    "judge_tokens",
    "judge_cost_usd",
]


def list_runs(path: Path | None = None) -> list[dict]:
    """All stored runs, newest first."""
    con = _connect(path)
    try:
        rows = con.execute(
            "SELECT run_id, suite, target, judge_model, mean_score, n_cases, created_at,"
            " n_errors, n_skipped, judge_tokens, judge_cost_usd"
            " FROM runs ORDER BY created_at DESC"
        ).fetchall()
    finally:
        con.close()
    cols = _RUN_META_COLUMNS
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
            "SELECT run_id, suite, target, judge_model, mean_score, n_cases, created_at,"
            " n_errors, n_skipped, judge_tokens, judge_cost_usd"
            " FROM runs WHERE run_id = ?",
            [run_id],
        ).fetchone()
    finally:
        con.close()
    if row is None:
        raise StoreError(f"unknown run id: {run_id!r}")
    cols = _RUN_META_COLUMNS
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
            " judge_reason, checks_json, error, judge_prompt_tokens, judge_completion_tokens,"
            " judge_cost_usd FROM case_results WHERE run_id = ?",
            [run_id],
        ).fetchall()
    finally:
        con.close()
    cols = ["case_id", "score", "judge_score", "latency_ms", "question", "answer", "judge_reason"]
    details = []
    for row in rows:
        detail = dict(zip(cols, row[:7]))
        detail["checks"] = json.loads(row[7]) if row[7] else None
        detail["error"] = row[8]
        detail["judge_prompt_tokens"] = row[9]
        detail["judge_completion_tokens"] = row[10]
        detail["judge_cost_usd"] = row[11]
        details.append(detail)
    return details


class CaseDelta(BaseModel):
    case_id: str
    score_a: float
    score_b: float
    delta: float
    latency_a: float | None = None  # ms
    latency_b: float | None = None
    cost_a: float | None = None  # judge USD; None: not judged or not priced
    cost_b: float | None = None


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
    # Over the shared cases only, so the two sides are like for like.
    mean_latency_a: float | None = None
    mean_latency_b: float | None = None
    judge_cost_a: float | None = None  # None when no shared case was priced
    judge_cost_b: float | None = None


def count_error_cases(run_id: str, path: Path | None = None) -> int:
    """Number of a run's cases stored with an error.

    Counted from ``case_results`` rather than ``runs.n_errors``, so runs
    saved before the per-run counts existed are checked too.
    """
    con = _connect(path)
    try:
        row = con.execute(
            "SELECT count(*) FROM case_results WHERE run_id = ? AND error IS NOT NULL",
            [run_id],
        ).fetchone()
    finally:
        con.close()
    return row[0]


def judged_case_ids(run_id: str, path: Path | None = None) -> set[str]:
    """Case ids that carry a judge score in a stored run."""
    con = _connect(path)
    try:
        rows = con.execute(
            "SELECT case_id FROM case_results WHERE run_id = ? AND judge_score IS NOT NULL",
            [run_id],
        ).fetchall()
    finally:
        con.close()
    return {row[0] for row in rows}


def compare_runs(ref_a: str, ref_b: str, path: Path | None = None) -> RunComparison:
    """Compare two stored runs case by case (inner join on case id)."""
    run_a = resolve_run_id(ref_a, path)
    run_b = resolve_run_id(ref_b, path)
    meta_a = load_run_meta(run_a, path)
    meta_b = load_run_meta(run_b, path)
    rows_a = _load_case_metrics(run_a, path)
    rows_b = _load_case_metrics(run_b, path)

    cases = [
        CaseDelta(
            case_id=c,
            score_a=rows_a[c][0],
            score_b=rows_b[c][0],
            delta=rows_b[c][0] - rows_a[c][0],
            latency_a=rows_a[c][1],
            latency_b=rows_b[c][1],
            cost_a=rows_a[c][2],
            cost_b=rows_b[c][2],
        )
        for c in sorted(set(rows_a) & set(rows_b))
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
        mean_latency_a=_mean([c.latency_a for c in cases]),
        mean_latency_b=_mean([c.latency_b for c in cases]),
        judge_cost_a=_known_sum([c.cost_a for c in cases]),
        judge_cost_b=_known_sum([c.cost_b for c in cases]),
    )


def _mean(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None


def _known_sum(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def _load_case_metrics(
    run_id: str, path: Path | None = None
) -> dict[str, tuple[float, float | None, float | None]]:
    """case_id -> (score, latency_ms, judge_cost_usd) for one run."""
    con = _connect(path)
    try:
        rows = con.execute(
            "SELECT case_id, score, latency_ms, judge_cost_usd FROM case_results"
            " WHERE run_id = ?",
            [run_id],
        ).fetchall()
    finally:
        con.close()
    return {case_id: (score, latency, cost) for case_id, score, latency, cost in rows}


# --- Baselines and the per-case regression gate (M5) -----------------------
#
# A baseline is a named pointer at a stored run: "the last run we trusted".
# The regression gate then asks a stricter question than the mean gate --
# not "is the average still fine?" but "did ANY single case fall past the
# threshold?", because a mean can improve while one case craters.


def pin_baseline(
    name: str, ref: str, path: Path | None = None, allow_errors: bool = False
) -> dict:
    """Point baseline ``name`` at the run ``ref`` resolves to.

    Re-pinning an existing name moves it to the new run. Returns the
    baseline joined with the run's meta (mean, n_cases).

    A run with error cases is refused unless ``allow_errors`` is set: a
    baseline built on an outage scores the broken cases 0.0, so every later
    run looks fine against it and real regressions stay hidden.
    """
    run_id = resolve_run_id(ref, path)
    meta = load_run_meta(run_id, path)
    n_errors = count_error_cases(run_id, path)
    if n_errors and not allow_errors:
        raise StoreError(
            f"run {run_id} has {n_errors} error case(s); refusing to pin it as a baseline "
            "(a baseline built on an outage hides real regressions). "
            "Pass --allow-errors to pin it anyway."
        )
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
