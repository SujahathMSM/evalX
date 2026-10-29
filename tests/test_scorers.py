"""Tests for the code-based scorers."""

from evalx.schemas import Check, TestCase
from evalx.scorers import score_case, score_case_details, score_check


def test_exact_match():
    check = Check(type="exact_match", expected="hello")
    assert score_check(check, "hello") == 1.0
    assert score_check(check, " hello ") == 1.0  # surrounding whitespace ignored
    assert score_check(check, "hello!") == 0.0


def test_contains_case_insensitive():
    check = Check(type="contains", expected="$4.2M")
    assert score_check(check, "revenue was $4.2m last year") == 1.0
    assert score_check(check, "no figure here") == 0.0


def test_regex():
    check = Check(type="regex", expected=r"(?i)no information|don't know")
    assert score_check(check, "I have no INFORMATION on that") == 1.0
    assert score_check(check, "It was $5") == 0.0


def test_weighted_mean():
    case = TestCase(
        id="q",
        question="?",
        checks=[
            Check(type="contains", expected="warehouse", weight=1.0),
            Check(type="contains", expected="supply chain", weight=1.0),
        ],
    )
    score, per_check = score_case(case, "supply chain issues")
    assert per_check == [0.0, 1.0]
    assert score == 0.5


def test_no_checks_scores_zero():
    case = TestCase(id="q", question="?")
    score, per_check = score_case(case, "anything")
    assert score == 0.0
    assert per_check == []


def test_score_case_details():
    case = TestCase(
        id="q",
        question="?",
        checks=[
            Check(type="contains", expected="warehouse", weight=1.0),
            Check(type="contains", expected="supply chain", weight=2.0),
        ],
    )
    details = score_case_details(case, "supply chain issues")
    assert [(d.type, d.expected, d.weight, d.score) for d in details] == [
        ("contains", "warehouse", 1.0, 0.0),
        ("contains", "supply chain", 2.0, 1.0),
    ]
