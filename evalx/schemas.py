"""Pydantic schemas: the shapes of test cases, scores, and runs."""

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

CheckType = Literal[
    "exact_match",
    "contains",
    "not_contains",
    "regex",
    "numeric",
    "max_length",
    "json_schema",
]


class Check(BaseModel):
    """One verifiable expectation about an answer.

    ``expected`` means, by type:

    - ``exact_match`` / ``contains`` / ``not_contains``: the text
    - ``regex``: a Python regular expression
    - ``numeric``: the number (pair with ``tolerance``, default 0)
    - ``max_length``: the most characters the answer may have
    - ``json_schema``: a JSON Schema the answer must validate against
      (a YAML mapping or a JSON string)

    Everything is validated when the dataset loads, so a typo in a check
    fails fast instead of silently scoring 0.
    """

    type: CheckType
    expected: str
    weight: float = Field(default=1.0, gt=0)
    tolerance: float | None = Field(default=None, ge=0)  # numeric only

    @field_validator("expected", mode="before")
    @classmethod
    def _schema_mapping_to_json(cls, value):
        # Let datasets write a json_schema check as a YAML mapping.
        if isinstance(value, dict):
            return json.dumps(value, sort_keys=True)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return str(value)  # numeric / max_length written as bare numbers
        return value

    @model_validator(mode="after")
    def _expected_fits_type(self):
        if self.tolerance is not None and self.type != "numeric":
            raise ValueError("tolerance only applies to numeric checks")
        if self.type == "regex":
            try:
                re.compile(self.expected)
            except re.error as exc:
                raise ValueError(f"invalid regex {self.expected!r}: {exc}") from None
        elif self.type == "numeric":
            try:
                float(self.expected.replace(",", ""))
            except ValueError:
                raise ValueError(f"numeric check needs a number, got {self.expected!r}") from None
        elif self.type == "max_length":
            if not self.expected.strip().isdigit() or int(self.expected) < 1:
                raise ValueError(
                    f"max_length check needs a positive integer, got {self.expected!r}"
                )
        elif self.type == "json_schema":
            from jsonschema import Draft202012Validator
            from jsonschema.exceptions import SchemaError

            try:
                schema = json.loads(self.expected)
                Draft202012Validator.check_schema(schema)
            except (ValueError, SchemaError) as exc:
                raise ValueError(f"invalid JSON Schema: {str(exc).splitlines()[0]}") from None
        return self


class JudgeSpec(BaseModel):
    """Rubric for the LLM judge on one case."""

    rubric: str
    weight: float = Field(default=1.0, gt=0)


class TestCase(BaseModel):
    """One golden test case: a question plus its answer key."""

    __test__ = False  # not a pytest test class; silence collection warning

    id: str
    question: str
    checks: list[Check] = []
    judge: JudgeSpec | None = None
    tags: list[str] = []
    difficulty: Literal["easy", "medium", "hard"] = "medium"


class CheckScore(BaseModel):
    """One check's outcome on one answer, with the expectation it graded."""

    type: str
    expected: str
    weight: float
    score: float


class CaseResult(BaseModel):
    """What happened when one case ran."""

    case_id: str
    question: str
    answer: str
    check_scores: list[float]
    checks: list[CheckScore] = []
    judge_score: float | None = None
    judge_reason: str | None = None
    score: float
    latency_ms: float
    # Set when the target or the judge raised on this case. The case still
    # counts in the mean with score 0.0, so a crashing app can't pass a gate.
    error: str | None = None
    # What the LLM judge consumed on this case (retries included). None when
    # the judge wasn't called, or the provider/model didn't report it.
    judge_prompt_tokens: int | None = None
    judge_completion_tokens: int | None = None
    judge_cost_usd: float | None = None
    # One score per attempt when the run used --repeats; empty for older
    # runs. ``score`` is their mean.
    attempt_scores: list[float] = []

    @property
    def spread(self) -> float | None:
        """Max minus min attempt score: how much the case moved between
        identical attempts. None with fewer than two attempts."""
        if len(self.attempt_scores) < 2:
            return None
        return max(self.attempt_scores) - min(self.attempt_scores)


class RunResult(BaseModel):
    """The outcome of a full suite run."""

    suite: str
    target: str
    n_cases: int
    mean_score: float
    results: list[CaseResult]
    # Cases that only have a judge rubric, run without a judge: nothing could
    # score them, so they are left out of ``results`` and the mean.
    skipped: list[str] = []
    # Wall-clock time of the whole run and the concurrency it ran with: mean
    # per-case latency rises with --workers even when the run gets faster.
    duration_s: float | None = None
    workers: int | None = None
    repeats: int | None = None  # attempts per case (--repeats)
    run_id: str | None = None  # assigned when the run is saved to the store
