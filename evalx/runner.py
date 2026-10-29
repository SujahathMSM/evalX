"""The runner: execute every case against the target and score the answers."""

import time
from pathlib import Path

from evalx.adapters import resolve_target
from evalx.datasets import load_dataset
from evalx.schemas import CaseResult, RunResult, TestCase
from evalx.scorers import score_case, score_case_details


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


def run_suite(suite: str | Path, target: str, judge=None) -> RunResult:
    """Run all cases in a dataset against a target app. Returns the run result.

    ``judge`` is optional and duck-typed: any object with
    ``.score(case, answer)`` returning a verdict with ``.score`` and ``.reason``.
    It is only consulted for cases that define a judge rubric.
    """
    cases = load_dataset(suite)
    answer_fn = resolve_target(target)

    results: list[CaseResult] = []
    for case in cases:
        started = time.perf_counter()
        answer = answer_fn(case.question)
        latency_ms = (time.perf_counter() - started) * 1000.0
        code_score, per_check = score_case(case, answer)

        judge_score: float | None = None
        judge_reason: str | None = None
        if judge is not None and case.judge is not None:
            verdict = judge.score(case, answer)
            judge_score, judge_reason = verdict.score, verdict.reason

        results.append(
            CaseResult(
                case_id=case.id,
                question=case.question,
                answer=answer,
                check_scores=per_check,
                checks=score_case_details(case, answer),
                judge_score=judge_score,
                judge_reason=judge_reason,
                score=_combine_score(case, code_score, judge_score),
                latency_ms=latency_ms,
            )
        )

    mean_score = sum(r.score for r in results) / len(results) if results else 0.0
    return RunResult(
        suite=str(suite),
        target=target,
        n_cases=len(results),
        mean_score=mean_score,
        results=results,
    )
