"""Re-grade a stored run's answers with the judge only: measure judge noise.

A case's score can move for two reasons: the app answered differently, or
the judge graded the same answer differently. ``--repeats`` moves both at
once. Rejudging freezes the answers (the app is never called) and asks the
judge again, N times per case, so whatever still moves is the judge alone.

With the run's own judge model that measures how consistent the judge is.
With another ``--judge-model`` it shows how a different judge would grade
the very same answers.

What is reused from the original run, per case: the stored answer and the
stored code-check results, so only the judge's verdict can change. The
rubric comes from the dataset as it is now (runs don't store rubrics).
"""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

from evalx.datasets import load_dataset
from evalx.runner import _aggregate, score_answer
from evalx.schemas import CaseResult, CheckScore, RunResult, TestCase
from evalx.store import load_case_details, load_run_meta, resolve_run_id


class RejudgeError(ValueError):
    """The run can't be rejudged as asked."""


class SkippedCase(BaseModel):
    case_id: str
    reason: str


class RejudgedCase(BaseModel):
    case_id: str
    original_judge: float | None  # the run's judge score (a mean, for --repeats runs)
    judge_scores: list[float]  # one per re-grade (failed re-grades left out)
    spread: float | None  # judge-only spread; None with fewer than 2 scores


class RejudgeResult(BaseModel):
    source_run_id: str
    source_repeats: int
    suite: str
    judge_model: str | None
    repeats: int
    result: RunResult  # rubric cases re-graded, all other cases carried over
    rejudged: list[RejudgedCase]
    skipped: list[SkippedCase]
    question_changed: list[str]  # cases whose dataset question differs from the stored one


def _stored_checks(detail: dict) -> list[CheckScore]:
    return [CheckScore(**c) for c in detail["checks"] or []]


def _carried(detail: dict) -> CaseResult:
    """A case copied unchanged from the source run (not re-graded)."""
    return CaseResult(
        case_id=detail["case_id"],
        question=detail["question"] or "",
        answer=detail["answer"] or "",
        check_scores=[c["score"] for c in detail["checks"] or []],
        checks=_stored_checks(detail),
        judge_score=detail["judge_score"],
        judge_reason=detail["judge_reason"],
        score=detail["score"],
        latency_ms=detail["latency_ms"] or 0.0,
        error=detail["error"],
        attempt_scores=detail["attempt_scores"] or [detail["score"]],
    )


def rejudge_run(
    ref: str,
    judge,
    repeats: int = 3,
    workers: int = 1,
    suite: str | Path | None = None,
    judge_model: str | None = None,
    path=None,
) -> RejudgeResult:
    """Re-grade the stored answers of run ``ref`` with ``judge``, ``repeats`` times.

    ``judge`` is anything with ``.score(case, answer)`` (see runner.run_suite).
    ``suite`` overrides the dataset path the run recorded, for when it moved.
    """
    if repeats < 1:
        raise RejudgeError(f"repeats must be >= 1, got {repeats}")
    if workers < 1:
        raise RejudgeError(f"workers must be >= 1, got {workers}")
    run_id = resolve_run_id(ref, path)
    meta = load_run_meta(run_id, path)
    suite_path = str(suite) if suite is not None else meta["suite"]
    try:
        cases = {c.id: c for c in load_dataset(suite_path)}
    except Exception as exc:  # DatasetError, or a path that no longer exists
        raise RejudgeError(
            f"cannot load the dataset for run {run_id} from {suite_path!r} ({exc}); "
            "pass --suite with its current location"
        ) from exc

    details = load_case_details(run_id, path)
    jobs: list[tuple[TestCase, dict]] = []
    skipped: list[SkippedCase] = []
    question_changed: list[str] = []
    for d in details:
        case = cases.get(d["case_id"])
        reason = None
        if d["answer"] is None:
            reason = "no stored answer (run saved before evalX kept answers)"
        elif d["error"] is not None:
            reason = "errored in the original run"
        elif case is None:
            reason = "no longer in the dataset"
        elif case.judge is None:
            reason = "no rubric"
        if reason:
            skipped.append(SkippedCase(case_id=d["case_id"], reason=reason))
            continue
        if d["question"] is not None and d["question"] != case.question:
            question_changed.append(d["case_id"])
        jobs.append((case, d))

    if not jobs:
        raise RejudgeError(
            f"run {run_id} has no case to rejudge: every case was skipped "
            f"({', '.join(sorted({s.reason for s in skipped})) or 'no cases'})"
        )

    def grade(job: tuple[TestCase, dict]) -> CaseResult:
        case, d = job
        return score_answer(
            case, d["answer"], d["latency_ms"] or 0.0, judge, details=_stored_checks(d)
        )

    started = time.perf_counter()
    flat = [job for job in jobs for _ in range(repeats)]
    if workers == 1 or len(flat) <= 1:
        attempts = [grade(job) for job in flat]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="evalx") as pool:
            attempts = list(pool.map(grade, flat))

    regraded: dict[str, CaseResult] = {}
    rejudged: list[RejudgedCase] = []
    for i, (case, d) in enumerate(jobs):
        group = attempts[i * repeats : (i + 1) * repeats]
        merged = _aggregate(group)
        # The answer never changed, so the attempts differ only in the judge.
        regraded[case.id] = merged
        scores = [a.judge_score for a in group if a.judge_score is not None]
        rejudged.append(
            RejudgedCase(
                case_id=case.id,
                original_judge=d["judge_score"],
                judge_scores=scores,
                spread=round(max(scores) - min(scores), 6) if len(scores) >= 2 else None,
            )
        )

    # Keep every case of the source run, in its order, so `evalx compare` on
    # the two runs lines up: re-graded ones replaced, the rest carried over.
    results = [regraded.get(d["case_id"]) or _carried(d) for d in details]
    mean = sum(r.score for r in results) / len(results) if results else 0.0
    result = RunResult(
        suite=meta["suite"],
        target=meta["target"],
        n_cases=len(results),
        mean_score=mean,
        results=results,
        duration_s=time.perf_counter() - started,
        workers=workers,
        repeats=repeats,
    )
    return RejudgeResult(
        source_run_id=run_id,
        source_repeats=meta["repeats"] or 1,
        suite=suite_path,
        judge_model=judge_model,
        repeats=repeats,
        result=result,
        rejudged=rejudged,
        skipped=skipped,
        question_changed=question_changed,
    )
