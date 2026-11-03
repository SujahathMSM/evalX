"""Tests for the code-based scorers."""

import pytest
from pydantic import ValidationError

from evalx.datasets import DatasetError, load_dataset
from evalx.schemas import Check, TestCase
from evalx.scorers import numbers_in, score_case, score_case_details, score_check


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


# --- not_contains, numeric, max_length, json_schema --------------------------



def test_not_contains_case_insensitive():
    check = Check(type="not_contains", expected="Acme Corp")
    assert score_check(check, "The supplier was Globex.") == 1.0
    assert score_check(check, "the supplier was ACME CORP.") == 0.0


@pytest.mark.parametrize(
    ("text", "numbers"),
    [
        ("torque to 40 in-lb", [40.0]),
        ("40psi and 12V", [40.0, 12.0]),
        ("-3 and 4.2", [-3.0, 4.2]),
        ("1,200.50 lbs", [1200.5]),
        ("1,2,3", [1.0, 2.0, 3.0]),
        (".5 inch", [0.5]),
        ("3.5mm", [3.5]),
        ("version v2, cell a1", []),
        ("1.2.3", []),
    ],
)
def test_numbers_in(text, numbers):
    assert numbers_in(text) == numbers


def test_numeric_exact_and_tolerance():
    exact = Check(type="numeric", expected="40")
    assert score_check(exact, "Torque the bolt to 40 in-lb.") == 1.0
    assert score_check(exact, "Torque the bolt to 41 in-lb.") == 0.0
    loose = Check(type="numeric", expected="40", tolerance=2)
    assert score_check(loose, "about 41.5 in-lb") == 1.0
    assert score_check(loose, "about 42 in-lb") == 1.0  # boundary is inclusive
    assert score_check(loose, "about 42.5 in-lb") == 0.0
    assert score_check(loose, "no number at all") == 0.0


def test_numeric_matches_any_number_in_answer():
    check = Check(type="numeric", expected="1200")
    assert score_check(check, "Between 2 and 3 cycles, at 1,200 rpm.") == 1.0


def test_numeric_expected_may_be_a_bare_yaml_number():
    assert Check(type="numeric", expected=4.2).expected == "4.2"
    assert Check(type="max_length", expected=280).expected == "280"


def test_max_length():
    check = Check(type="max_length", expected="10")
    assert score_check(check, "  short  ") == 1.0  # surrounding whitespace ignored
    assert score_check(check, "exactly 10") == 1.0
    assert score_check(check, "eleven char") == 0.0


SCHEMA = {
    "type": "object",
    "required": ["part", "torque"],
    "properties": {"part": {"type": "string"}, "torque": {"type": "number"}},
}


def test_json_schema_valid_and_invalid():
    check = Check(type="json_schema", expected=SCHEMA)  # a mapping, as YAML gives it
    assert score_check(check, '{"part": "AN3", "torque": 25}') == 1.0
    assert score_check(check, '{"part": "AN3"}') == 0.0  # missing required key
    assert score_check(check, '{"part": "AN3", "torque": "25"}') == 0.0  # wrong type
    assert score_check(check, "Sure! Here it is: part AN3") == 0.0  # not JSON


def test_json_schema_allows_one_code_fence():
    check = Check(type="json_schema", expected=SCHEMA)
    fenced = '```json\n{"part": "AN3", "torque": 25}\n```'
    assert score_check(check, fenced) == 1.0
    assert score_check(check, '```\n{"part": "AN3", "torque": 25}\n```') == 1.0


def test_json_schema_accepts_a_json_string():
    import json

    check = Check(type="json_schema", expected=json.dumps(SCHEMA))
    assert score_check(check, '{"part": "AN3", "torque": 25}') == 1.0


# --- validation: a bad check fails at load time, not silently at score time --


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"type": "regex", "expected": "([unclosed"}, "invalid regex"),
        ({"type": "numeric", "expected": "forty"}, "needs a number"),
        ({"type": "max_length", "expected": "0"}, "positive integer"),
        ({"type": "max_length", "expected": "ten"}, "positive integer"),
        ({"type": "json_schema", "expected": "{not json"}, "invalid JSON Schema"),
        ({"type": "json_schema", "expected": {"type": "nope"}}, "invalid JSON Schema"),
        ({"type": "contains", "expected": "x", "tolerance": 1}, "tolerance only applies"),
        ({"type": "numeric", "expected": "4", "tolerance": -1}, "greater than or equal"),
        ({"type": "fuzzy", "expected": "x"}, "Input should be"),
    ],
)
def test_bad_checks_are_rejected(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        Check(**kwargs)


def test_dataset_with_bad_check_fails_to_load(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text(
        "- id: c1\n"
        "  question: q\n"
        "  checks:\n"
        "    - {type: numeric, expected: forty}\n"
    )
    with pytest.raises(DatasetError, match="needs a number"):
        load_dataset(path)


def test_dataset_with_new_check_types_loads(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text(
        "- id: c1\n"
        "  question: q\n"
        "  checks:\n"
        "    - {type: numeric, expected: 40, tolerance: 2}\n"
        "    - {type: not_contains, expected: Acme}\n"
        "    - {type: max_length, expected: 280}\n"
        "    - type: json_schema\n"
        "      expected:\n"
        "        type: object\n"
        "        required: [torque]\n"
    )
    (case,) = load_dataset(path)
    assert [c.type for c in case.checks] == ["numeric", "not_contains", "max_length", "json_schema"]
    assert case.checks[0].tolerance == 2


def test_new_check_types_through_a_full_run(tmp_path):
    from evalx.runner import run_suite

    path = tmp_path / "cases.yaml"
    path.write_text(
        "- id: revenue\n"
        "  question: \"What was the company's revenue in 2024?\"\n"
        "  checks:\n"
        "    - {type: numeric, expected: 4.2, tolerance: 0.05}\n"
        "    - {type: not_contains, expected: '3.8M'}\n"  # the 2023 figure
        "    - {type: max_length, expected: 100}\n"
        "- id: ceo-json\n"
        "  question: Who is the CEO? Reply as JSON.\n"
        "  checks:\n"
        "    - type: json_schema\n"
        "      expected: {type: object, required: [ceo]}\n"
    )
    result = run_suite(path, "examples.demo_app:answer")
    by_id = {r.case_id: r for r in result.results}
    assert by_id["revenue"].check_scores == [1.0, 1.0, 1.0]
    assert by_id["ceo-json"].score == 0.0  # the demo app answers in prose
    assert [c.type for c in by_id["revenue"].checks] == ["numeric", "not_contains", "max_length"]
