"""The runner: execute every case against the target and score the answers."""

import time
from pathlib import Path

from evalx.adapters import resolve_target
from evalx.datasets import load_dataset
from evalx.schemas import CaseResult, RunResult
from evalx.scorers import score_case


def run_suite(suite: str | Path, target: str) -> RunResult:
    """Run all cases in a dataset against a target app. Returns the run result."""
    cases = load_dataset(suite)
    answer_fn = resolve_target(target)

    results: list[CaseResult] = []
    for case in cases:
        started = time.perf_counter()
        answer = answer_fn(case.question)
        latency_ms = (time.perf_counter() - started) * 1000.0
        score, per_check = score_case(case, answer)
        results.append(
            CaseResult(
                case_id=case.id,
                question=case.question,
                answer=answer,
                check_scores=per_check,
                score=score,
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
