"""Code-based scorers: deterministic checks, each returning 0.0..1.0."""

import re

from evalx.schemas import Check, CheckScore, TestCase


def _exact_match(check: Check, answer: str) -> float:
    return 1.0 if answer.strip() == check.expected.strip() else 0.0


def _contains(check: Check, answer: str) -> float:
    return 1.0 if check.expected.lower() in answer.lower() else 0.0


def _regex(check: Check, answer: str) -> float:
    return 1.0 if re.search(check.expected, answer) else 0.0


_SCORERS = {
    "exact_match": _exact_match,
    "contains": _contains,
    "regex": _regex,
}


def score_check(check: Check, answer: str) -> float:
    """Score one check against an answer."""
    return _SCORERS[check.type](check, answer)


def score_case_details(case: TestCase, answer: str) -> list[CheckScore]:
    """Per-check detail for the record: what was expected and how it scored."""
    return [
        CheckScore(type=c.type, expected=c.expected, weight=c.weight, score=score_check(c, answer))
        for c in case.checks
    ]


def combine_checks(details: list[CheckScore]) -> float:
    """Weighted mean of already-scored checks (0.0 when there are none)."""
    total_weight = sum(d.weight for d in details)
    if not total_weight:
        return 0.0
    return sum(d.score * d.weight for d in details) / total_weight


def score_case(case: TestCase, answer: str) -> tuple[float, list[float]]:
    """Weighted mean of a case's checks. Returns (score, per-check scores)."""
    details = score_case_details(case, answer)
    return combine_checks(details), [d.score for d in details]
