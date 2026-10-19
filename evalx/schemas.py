"""Pydantic schemas: the shapes of test cases, scores, and runs."""

from typing import Literal

from pydantic import BaseModel, Field


class Check(BaseModel):
    """One verifiable expectation about an answer."""

    type: Literal["exact_match", "contains", "regex"]
    expected: str
    weight: float = Field(default=1.0, gt=0)


class TestCase(BaseModel):
    """One golden test case: a question plus its answer key."""

    id: str
    question: str
    checks: list[Check] = []
    tags: list[str] = []
    difficulty: Literal["easy", "medium", "hard"] = "medium"


class CaseResult(BaseModel):
    """What happened when one case ran."""

    case_id: str
    question: str
    answer: str
    check_scores: list[float]
    score: float
    latency_ms: float


class RunResult(BaseModel):
    """The outcome of a full suite run."""

    suite: str
    target: str
    n_cases: int
    mean_score: float
    results: list[CaseResult]
