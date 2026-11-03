"""The runner: execute every case against the target and score the answers."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evalx.adapters import resolve_target
from evalx.datasets import load_dataset
from evalx.schemas import CaseResult, RunResult, TestCase
from evalx.scorers import combine_checks, score_case_details


def _combine_score(case: TestCase, code_score: float, judge_score: float | None) -> float:
    """Weighted mean of code checks and the judge verdict by their weights."""
    parts: list[tuple[float, float]] = []
    if case.checks:
        parts.append((code_score, sum(c.weight for c in case.checks)))
    if judge_score is not None and case.judge is not None:
        parts.append((judge_score, case.judge.weight))
    if not parts:
        return 0.0
    total_weight = sum(w for _, w in parts)
    return sum(s * w for s, w in parts) / total_weight


def _error_text(stage: str, exc: Exception) -> str:
    return f"{stage}: {type(exc).__name__}: {exc}"


def _run_case(case: TestCase, answer_fn, judge) -> CaseResult:
    """Ask the target one case, score it, and never raise (errors are recorded)."""
    started = time.perf_counter()
    try:
        answer = answer_fn(case.question)
    except Exception as exc:  # noqa: BLE001 - any app failure is a case failure
        return CaseResult(
            case_id=case.id,
            question=case.question,
            answer="",
            check_scores=[],
            score=0.0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=_error_text("target", exc),
        )
    latency_ms = (time.perf_counter() - started) * 1000.0
    if not isinstance(answer, str):
        answer = "" if answer is None else str(answer)

    details = score_case_details(case, answer)
    code_score = combine_checks(details)

    judge_score: float | None = None
    judge_reason: str | None = None
    error: str | None = None
    usage = None
    if judge is not None and case.judge is not None:
        try:
            verdict = judge.score(case, answer)
            judge_score, judge_reason = verdict.score, verdict.reason
            usage = getattr(verdict, "usage", None)
        except Exception as exc:  # noqa: BLE001 - recorded on the case
            error = _error_text("judge", exc)
            usage = getattr(exc, "usage", None)  # failed attempts still cost

    return CaseResult(
        case_id=case.id,
        question=case.question,
        answer=answer,
        check_scores=[d.score for d in details],
        checks=details,
        judge_score=judge_score,
        judge_reason=judge_reason,
        # A failed judge leaves the case unscored: 0.0, not a code-only
        # score that would quietly drop the rubric from the grade.
        score=0.0 if error else _combine_score(case, code_score, judge_score),
        latency_ms=latency_ms,
        error=error,
        judge_prompt_tokens=getattr(usage, "prompt_tokens", None),
        judge_completion_tokens=getattr(usage, "completion_tokens", None),
        judge_cost_usd=getattr(usage, "cost_usd", None),
    )


def run_suite(suite: str | Path, target: str, judge=None, workers: int = 1) -> RunResult:
    """Run all cases in a dataset against a target app. Returns the run result.

    ``judge`` is optional and duck-typed: any object with
    ``.score(case, answer)`` returning a verdict with ``.score`` and ``.reason``.
    It is only consulted for cases that define a judge rubric.

    One failing case never aborts the run: if the target or the judge raises,
    the case is recorded with ``error`` set and a score of 0.0. Cases that
    have only a judge rubric and no checks are skipped (listed in
    ``RunResult.skipped``, left out of the mean) when no judge is given.

    ``workers`` > 1 runs cases concurrently on a thread pool (target and judge
    calls are I/O-bound, so threads are enough). The target and judge must
    then be safe to call from several threads. Results keep dataset order
    whatever order the cases finish in, so a parallel run reports exactly
    like a sequential one.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    cases = load_dataset(suite)
    answer_fn = resolve_target(target)

    started = time.perf_counter()
    skipped = [c.id for c in cases if not c.checks and judge is None]
    to_run = [c for c in cases if c.checks or judge is not None]

    if workers == 1 or len(to_run) <= 1:
        results = [_run_case(case, answer_fn, judge) for case in to_run]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="evalx") as pool:
            results = list(pool.map(lambda case: _run_case(case, answer_fn, judge), to_run))

    mean_score = sum(r.score for r in results) / len(results) if results else 0.0
    return RunResult(
        suite=str(suite),
        target=target,
        n_cases=len(results),
        mean_score=mean_score,
        results=results,
        skipped=skipped,
        duration_s=time.perf_counter() - started,
        workers=workers,
    )
