"""Code-based scorers: deterministic checks, each returning 0.0..1.0."""

import json
import re

from jsonschema import Draft202012Validator

from evalx.schemas import Check, CheckScore, TestCase


def _exact_match(check: Check, answer: str) -> float:
    return 1.0 if answer.strip() == check.expected.strip() else 0.0


def _contains(check: Check, answer: str) -> float:
    return 1.0 if check.expected.lower() in answer.lower() else 0.0


def _not_contains(check: Check, answer: str) -> float:
    """1.0 when the text is absent (case-insensitive): a banned phrase, a
    hallucinated name, a leaked secret."""
    return 0.0 if check.expected.lower() in answer.lower() else 1.0


def _regex(check: Check, answer: str) -> float:
    return 1.0 if re.search(check.expected, answer) else 0.0


# A run of digits, commas and dots, optionally signed. Units may follow
# directly ("40psi", "3.5mm"); a number glued to letters before it ("v2",
# "a1") is an identifier, not a quantity.
_NUMBER_TOKEN = re.compile(r"(?<![\w.,])-?[\d.,]*\d")
_THOUSANDS = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?")


def numbers_in(text: str) -> list[float]:
    """Every number written in the text, in order.

    "1,200" and "1,200.50" read as thousands; any other comma separates
    numbers ("1,2,3" is 1, 2 and 3).
    """
    found = []
    for match in _NUMBER_TOKEN.finditer(text):
        token = match.group()
        parts = [token] if _THOUSANDS.fullmatch(token) else token.split(",")
        for part in parts:
            try:
                found.append(float(part.replace(",", "")))
            except ValueError:
                continue  # e.g. "1.2.3"
    return found


def _numeric(check: Check, answer: str) -> float:
    """1.0 when any number in the answer is within tolerance of expected."""
    target = float(check.expected.replace(",", ""))
    tolerance = check.tolerance or 0.0
    return 1.0 if any(abs(n - target) <= tolerance + 1e-12 for n in numbers_in(answer)) else 0.0


def _max_length(check: Check, answer: str) -> float:
    return 1.0 if len(answer.strip()) <= int(check.expected) else 0.0


_FENCE = re.compile(r"^\s*```[\w-]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)


def _json_schema(check: Check, answer: str) -> float:
    """1.0 when the answer is JSON that validates against the schema.

    A single Markdown code fence around the JSON (```json ... ```) is
    allowed, since chat models add one unasked.
    """
    text = answer.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text)
    except ValueError:
        return 0.0
    validator = Draft202012Validator(json.loads(check.expected))
    return 1.0 if validator.is_valid(data) else 0.0


_SCORERS = {
    "exact_match": _exact_match,
    "contains": _contains,
    "not_contains": _not_contains,
    "regex": _regex,
    "numeric": _numeric,
    "max_length": _max_length,
    "json_schema": _json_schema,
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
